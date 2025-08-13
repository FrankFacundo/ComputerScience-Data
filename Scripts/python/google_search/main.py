import os

from google_search import GoogleCustomSearchClient

api_key = os.getenv("GOOGLE_SEARCH_KEY")
cx = os.getenv("GOOGLE_CSE_ID")
client = GoogleCustomSearchClient(api_key=api_key, cx=cx)

# 2) Búsqueda web normal
res = client.search("prix de cartes", num=10, hl="fr", gl="FR", safe="off")
for item in res["items_simplified"]:
    print(item["title"], "->", item["link"])

# 3) Restringir a un sitio (equivale a site:bgl.lu)
res = client.search_site("rdv", site="bgl.lu", hl="fr", gl="LU")
for item in res["items_simplified"]:
    print(item["title"], "->", item["link"])

# 4) Paginación (máx 100 resultados por la API)
for item in client.iterate("open banking", pages=3, num=10, hl="en", gl="US"):
    print(item["title"])

# 5) Búsqueda de imágenes (con filtros)
imgs = client.search_images(
    "Luxembourg old town",
    num=5,
    imgSize="large",
    imgType="photo",
    imgColorType="color",
    dateRestrict=GoogleCustomSearchClient.date_restrict(weeks=2),
)
for im in imgs["items_simplified"]:
    print(im["title"], "->", im["image"], "(thumb:", im["thumbnail"], ")")
