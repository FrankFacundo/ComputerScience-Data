"""
Script based on documentation: 
https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
"""

import os
import time
from typing import Dict, Iterable, List, Optional, Union

import requests


class GoogleCustomSearchClient:
    """
    Cliente para Google Custom Search JSON API (cse.list).

    Uso rápido:
        client = GoogleCustomSearchClient.from_env()
        # Búsqueda web
        items = client.search("prix de cartes", num=10)["items_simplified"]
        # Búsqueda restringida a un sitio
        items = client.search_site("rdv", site="bgl.lu")["items_simplified"]
        # Iterar varias páginas (máx 100 resultados por la API)
        for it in client.iterate("open banking", pages=5, num=10):
            print(it["title"], it["link"])
        # Búsqueda de imágenes
        imgs = client.search_images("luxembourg city", num=5)["items_simplified"]

    Variables de entorno soportadas:
        - GOOGLE_API_KEY  : API key de Google
        - GOOGLE_CSE_ID   : ID del Programmable Search Engine (cx)
    """

    BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str,
        cx: str,
        timeout: int = 20,
        session: Optional[requests.Session] = None,
        retries: int = 3,
        backoff: float = 0.8,
    ) -> None:
        if not api_key:
            raise ValueError("api_key es obligatorio (o usa from_env()).")
        if not cx:
            raise ValueError("cx (Programmable Search Engine ID) es obligatorio.")
        self.api_key = api_key
        self.cx = cx
        self.timeout = timeout
        self.session = session or requests.Session()
        self.retries = max(0, retries)
        self.backoff = max(0.0, backoff)

    # ---------- Constructores ----------

    @classmethod
    def from_env(cls, *, timeout: int = 20, retries: int = 3, backoff: float = 0.8):
        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CSE_ID")
        return cls(
            api_key=api_key, cx=cx, timeout=timeout, retries=retries, backoff=backoff
        )

    # ---------- API pública principal ----------

    def search(
        self,
        q: str,
        *,
        num: int = 10,
        start: Optional[int] = None,
        searchType: Optional[str] = None,  # "image" para búsqueda de imágenes
        safe: Optional[str] = None,  # "active" | "off"
        filter: Optional[Union[str, int]] = None,  # "0" | "1"
        siteSearch: Optional[str] = None,
        siteSearchFilter: Optional[str] = None,  # "i" | "e"
        hl: Optional[str] = None,
        lr: Optional[str] = None,
        gl: Optional[str] = None,
        cr: Optional[str] = None,
        dateRestrict: Optional[str] = None,  # e.g., "d7", "m1", "y2"
        exactTerms: Optional[str] = None,
        excludeTerms: Optional[str] = None,
        orTerms: Optional[Union[str, List[str]]] = None,
        rights: Optional[str] = None,
        imgSize: Optional[
            str
        ] = None,  # "icon"|"small"|"medium"|"large"|"xlarge"|"xxlarge"|"huge"
        imgType: Optional[
            str
        ] = None,  # "clipart"|"face"|"lineart"|"stock"|"photo"|"animated"
        imgColorType: Optional[str] = None,  # "color"|"gray"|"mono"|"trans"
        imgDominantColor: Optional[str] = None,  # "red"|"blue"|...
        sort: Optional[str] = None,  # e.g., "date"
        linkSite: Optional[str] = None,
        lowRange: Optional[str] = None,
        highRange: Optional[str] = None,
        raw: bool = False,  # si True, devuelve el JSON completo de la API
    ) -> Dict:
        """
        Ejecuta una búsqueda única y devuelve dict con:
          - "raw": respuesta cruda (si raw=True)
          - "items_simplified": lista de resultados simplificados
          - "search_metadata": info útil (request, totalResults, next_start, etc.)
        """
        self._validate_num_start(num, start)
        if searchType is not None and searchType != "image":
            raise ValueError('searchType solo admite "image" o None.')
        if safe is not None and safe not in ("active", "off"):
            raise ValueError('safe debe ser "active" o "off".')
        if filter is not None:
            filter = str(filter)
            if filter not in ("0", "1"):
                raise ValueError('filter debe ser "0" o "1".')
        if siteSearchFilter is not None and siteSearchFilter not in ("i", "e"):
            raise ValueError('siteSearchFilter debe ser "i" (incluir) o "e" (excluir).')
        if isinstance(orTerms, list):
            orTerms = " ".join(orTerms)

        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": q,
            "num": num,
        }
        # Campos opcionales (solo si están definidos)
        opt = {
            "start": start,
            "searchType": searchType,
            "safe": safe,
            "filter": filter,
            "siteSearch": siteSearch,
            "siteSearchFilter": siteSearchFilter,
            "hl": hl,
            "lr": lr,
            "gl": gl,
            "cr": cr,
            "dateRestrict": dateRestrict,
            "exactTerms": exactTerms,
            "excludeTerms": excludeTerms,
            "orTerms": orTerms,
            "rights": rights,
            "imgSize": imgSize,
            "imgType": imgType,
            "imgColorType": imgColorType,
            "imgDominantColor": imgDominantColor,
            "sort": sort,
            "linkSite": linkSite,
            "lowRange": lowRange,
            "highRange": highRange,
        }
        params.update({k: v for k, v in opt.items() if v is not None})

        data = self._request(params)
        simplified = self._simplify_items(data.get("items", []), searchType=searchType)
        next_start = self._compute_next_start(data, current_start=start, num=num)

        result = {
            "items_simplified": simplified,
            "search_metadata": {
                "query": q,
                "totalResults": int(
                    data.get("searchInformation", {}).get("totalResults", "0") or 0
                ),
                "start": start or 1,
                "returned": len(simplified),
                "next_start": next_start,
                "searchType": searchType or "web",
            },
        }
        if raw:
            result["raw"] = data
        return result

    def iterate(
        self,
        q: str,
        *,
        pages: int = 10,
        num: int = 10,
        **kwargs,
    ) -> Iterable[Dict]:
        """
        Itera sobre resultados simplificados en múltiples páginas.
        Respeta el límite de la API (máx 100 resultados en total).
        """
        if not (1 <= num <= 10):
            raise ValueError("num debe estar entre 1 y 10.")
        # Ajuste de páginas máximo por límite de 100 resultados
        max_pages = min(pages, (100 + num - 1) // num)
        start = kwargs.pop("start", None) or 1
        total_yielded = 0

        for _ in range(max_pages):
            # No permitir sobrepasar 100 en start + num
            if start > 100:
                break
            if start + num - 1 > 100:
                num = 100 - start + 1
                if num <= 0:
                    break

            res = self.search(q, num=num, start=start, **kwargs)
            items = res["items_simplified"]
            for it in items:
                yield it
                total_yielded += 1
            if not items:
                break

            next_start = res["search_metadata"]["next_start"]
            if not next_start or next_start <= start:
                break
            start = next_start

            # Si ya llegamos a 100, parar
            if total_yielded >= 100:
                break

    # ---------- Helpers de conveniencia ----------

    def search_site(self, q: str, *, site: str, include: bool = True, **kwargs) -> Dict:
        """
        Busca restringiendo a un sitio (o excluyéndolo).
        Equivalente a siteSearch + siteSearchFilter.
        """
        return self.search(
            q,
            siteSearch=site,
            siteSearchFilter="i" if include else "e",
            **kwargs,
        )

    def search_images(self, q: str, **kwargs) -> Dict:
        """Atajo para búsqueda de imágenes (searchType='image')."""
        return self.search(q, searchType="image", **kwargs)

    @staticmethod
    def date_restrict(
        *, days: int = 0, weeks: int = 0, months: int = 0, years: int = 0
    ) -> str:
        """
        Construye el valor para dateRestrict: dN / wN / mN / yN (usa la unidad más grande no nula).
        """
        if years:
            return f"y{years}"
        if months:
            return f"m{months}"
        if weeks:
            return f"w{weeks}"
        if days:
            return f"d{days}"
        raise ValueError("Indica days|weeks|months|years > 0.")

    def set_cx(self, cx: str) -> None:
        """Permite cambiar el Programmable Search Engine ID en caliente."""
        if not cx:
            raise ValueError("cx no puede ser vacío.")
        self.cx = cx

    # ---------- Internos ----------

    def _request(self, params: Dict) -> Dict:
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(
                    self.BASE_URL, params=params, timeout=self.timeout
                )
                # Errores HTTP
                if not resp.ok:
                    # Intentar extraer detalle de error JSON si existe
                    try:
                        payload = resp.json()
                        if "error" in payload:
                            msg = payload["error"].get("message", str(payload))
                            raise requests.HTTPError(
                                f"{resp.status_code}: {msg}", response=resp
                            )
                    except ValueError:
                        pass
                    resp.raise_for_status()
                data = resp.json()
                # Errores de la API dentro de 200 OK
                if "error" in data:
                    raise requests.HTTPError(str(data["error"]))
                return data
            except (requests.RequestException, requests.HTTPError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
                else:
                    break
        assert last_err is not None
        raise last_err

    @staticmethod
    def _validate_num_start(num: int, start: Optional[int]) -> None:
        if not (1 <= num <= 10):
            raise ValueError("num debe estar entre 1 y 10.")
        if start is not None and start < 1:
            raise ValueError("start debe ser >= 1.")
        if start is not None and start + num - 1 > 100:
            # La API falla si start+num > 100
            raise ValueError(
                "La API no permite start + num > 100 (máximo 100 resultados)."
            )

    @staticmethod
    def _compute_next_start(
        data: Dict, current_start: Optional[int], num: int
    ) -> Optional[int]:
        """
        Calcula el siguiente índice 'start' para la paginación, si existe.
        """
        # Basado en "queries" -> "nextPage"
        queries = data.get("queries", {})
        next_pages = queries.get("nextPage", [])
        if next_pages:
            return next_pages[0].get("startIndex")
        # fallback básico
        if current_start is None:
            current_start = 1
        next_start = current_start + num
        total = int(data.get("searchInformation", {}).get("totalResults", "0") or 0)
        if next_start <= 100 and next_start <= total:
            return next_start
        return None

    @staticmethod
    def _simplify_items(items: List[Dict], *, searchType: Optional[str]) -> List[Dict]:
        simplified: List[Dict] = []
        for it in items:
            base = {
                "title": it.get("title"),
                "link": it.get("link"),
                "displayLink": it.get("displayLink"),
                "snippet": it.get("snippet"),
                "mime": it.get("mime"),
            }
            if searchType == "image":
                pagemap = it.get("pagemap", {})
                # Intentar extraer metadatos de imagen (cuando estén)
                imgobj = None
                if "cse_image" in pagemap and pagemap["cse_image"]:
                    imgobj = pagemap["cse_image"][0]
                image = {
                    "image": (imgobj or {}).get("src"),
                    "thumbnail": (pagemap.get("cse_thumbnail", [{}])[0] or {}).get(
                        "src"
                    ),
                }
                base.update(image)
            simplified.append(base)
        return simplified
