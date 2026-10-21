import json
import time
import urllib.parse
import urllib.request

import pandas as pd

# Reconstruye una muestra de la OBT de GrantMatch (PSet 2, OBT.OBT_AWARD_WORK)
# desde la API publica de OpenAlex, porque los trials de Snowflake del equipo
# expiraron. Misma fuente que el bucket S3 que carga Kestra y mismas reglas de
# limpieza de dbt Silver (slv_awards_history, slv_funders). Grain: un par
# award + work financiado (el de FACT_AWARD_WORKS).
API = "https://api.openalex.org"
MUESTRA_AWARDS = 4000
MAX_WORKS_POR_AWARD = 3  # tope para no gastar el presupuesto diario de la API
SEMILLA = 42
SALIDA = "obt_award_work.csv"


def pedir(ruta, **params):
    url = f"{API}/{ruta}?{urllib.parse.urlencode(params)}"
    for intento in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as respuesta:
                return json.load(respuesta)
        except Exception:
            time.sleep(2 ** intento)
    raise RuntimeError(f"Fallo la consulta: {url}")


def oa_id(valor):
    # Mismo criterio que el macro oa_id: de URL a ID corto.
    return valor.rstrip("/").split("/")[-1] if valor else None


def por_lotes(ids, ruta, filtro, campos):
    resultados = []
    for i in range(0, len(ids), 100):
        lote = "|".join(ids[i:i + 100])
        resultados += pedir(ruta, filter=f"{filtro}:{lote}", per_page=100, select=campos)["results"]
    return resultados


# 1. Awards: muestra aleatoria de los que tienen publicaciones enlazadas.
awards = []
for pagina in range(1, MUESTRA_AWARDS // 200 + 1):
    awards += pedir(
        "awards", sample=MUESTRA_AWARDS, seed=SEMILLA, per_page=200, page=pagina,
        filter="funded_outputs_count:>0",
    )["results"]
print(f"Awards descargados: {len(awards)}")

filas_award = []
pares = []
for a in awards:
    monto = a.get("amount")
    monto = float(monto) if monto is not None else None
    moneda = (a.get("currency") or "").strip().upper() or None
    inicio = pd.to_datetime(a.get("start_date"), errors="coerce")
    if pd.isna(inicio) and a.get("start_year") and 1900 <= a["start_year"] <= 2100:
        inicio = pd.Timestamp(a["start_year"], 1, 1)
    fin = pd.to_datetime(a.get("end_date"), errors="coerce")
    inconsistente = pd.notna(fin) and pd.notna(inicio) and fin < inicio
    if inconsistente:
        fin = pd.NaT
    topic = a.get("primary_topic") or {}
    instituciones = a.get("institution_awarded") or []
    lider = instituciones[0] if instituciones else {}
    filas_award.append({
        "award_key": oa_id(a["id"]),
        "funder_award_id": a.get("funder_award_id"),
        "award_title": a.get("display_name"),
        "has_description": bool((a.get("description") or "").strip()),
        "funder_scheme": a.get("funder_scheme"),
        "funding_type": (a.get("funding_type") or "").strip().lower() or None,
        "funder_key": oa_id((a.get("funder") or {}).get("id")),
        "funder_name": (a.get("funder") or {}).get("display_name"),
        "lead_institution_name": lider.get("display_name"),
        "lead_institution_country_code": lider.get("country_code"),
        "lead_institution_type": lider.get("type"),
        "award_topic_name": topic.get("display_name"),
        "award_field_name": (topic.get("field") or {}).get("display_name"),
        "award_domain_name": (topic.get("domain") or {}).get("display_name"),
        "award_start_date": inicio,
        "award_start_year": inicio.year if pd.notna(inicio) else None,
        # Reglas de slv_awards_history (A06, A08 y amount_usd solo en USD).
        "amount": monto if monto is not None and monto > 0 else None,
        "currency": moneda,
        "amount_usd": monto if moneda == "USD" and monto is not None and monto > 0 else None,
        "is_amount_invalid": monto is not None and monto <= 0,
        "has_inconsistent_dates": inconsistente,
        "duration_days": (fin - inicio).days if pd.notna(fin) and pd.notna(inicio) else None,
        "funded_outputs_count": a.get("funded_outputs_count") or 0,
        "institutions_count": len(instituciones),
        "provenance": a.get("provenance"),
    })
    for work in (a.get("funded_outputs") or [])[:MAX_WORKS_POR_AWARD]:
        pares.append((oa_id(a["id"]), oa_id(work)))

awards_df = pd.DataFrame(filas_award).drop_duplicates("award_key")

# 2. Funders: pais (slv_funders: country_code en mayusculas). La API no
# expone is_global_south, que si viene en el snapshot de S3.
funder_ids = sorted(awards_df["funder_key"].dropna().unique())
funders = por_lotes(funder_ids, "funders", "ids.openalex", "id,country_code")
funders_df = pd.DataFrame([{
    "funder_key": oa_id(f["id"]),
    "funder_country_code": (f.get("country_code") or "").upper() or None,
} for f in funders])
print(f"Funders descargados: {len(funders_df)}")

# 3. Works financiados por esos awards.
work_ids = sorted({w for _, w in pares})
campos_work = ("id,doi,title,abstract_inverted_index,publication_year,publication_date,type,"
               "language,cited_by_count,fwci,is_retracted,authorships,primary_topic")
works = por_lotes(work_ids, "works", "ids.openalex", campos_work)
works_df = pd.DataFrame([{
    "work_key": oa_id(w["id"]),
    "work_doi": w.get("doi"),
    "work_title": w.get("title"),
    "has_abstract": w.get("abstract_inverted_index") is not None,
    "publication_year": w.get("publication_year"),
    "publication_date": w.get("publication_date"),
    "work_type": w.get("type"),
    "language": w.get("language"),
    "cited_by_count": w.get("cited_by_count"),
    "fwci": w.get("fwci"),
    "is_retracted": w.get("is_retracted"),
    "authors_count": len(w.get("authorships") or []),
    "work_field_name": (((w.get("primary_topic") or {}).get("field")) or {}).get("display_name"),
    "work_domain_name": (((w.get("primary_topic") or {}).get("domain")) or {}).get("display_name"),
} for w in works])
print(f"Works descargados: {len(works_df)} de {len(work_ids)} pedidos")

# 4. OBT: hecho base (pares) + joins muchos a uno. Inner join con works,
# como Gold filtra a los works que si se cargaron.
obt = (
    pd.DataFrame(pares, columns=["award_key", "work_key"]).drop_duplicates()
    .merge(works_df, on="work_key", how="inner")
    .merge(awards_df, on="award_key", how="left")
    .merge(funders_df, on="funder_key", how="left")
)
obt.insert(2, "label", 1)
assert not obt.duplicated(["award_key", "work_key"]).any(), "Se rompio el grain"
obt.to_csv(SALIDA, index=False)
print(f"{SALIDA}: {obt.shape[0]} filas, {obt.shape[1]} columnas")
