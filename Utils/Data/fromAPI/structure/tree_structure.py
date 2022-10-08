class Aggregation(object):
    """
    Aggregation options:
        - country
        - region
        - adminregion
        - incomeLevel
        - lendingType

    Querying a country shows you which aggregate groups it belongs to:
    http://api.worldbank.org/v2/country/BRA

    COUNTRY: https://api.worldbank.org/v2/country?format=json&page=1
    INCOMELEVEL: https://api.worldbank.org/v2/incomeLevel?format=json
    LENDINGTYPE: https://api.worldbank.org/v2/lendingType?format=json
    REGION: https://api.worldbank.org/v2/region?format=json

    To check documentation of aggregation:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/898614-aggregate-api-queries
    """
    COUNTRY = "country"
    INCOMELEVEL = "incomeLevel"
    LENDINGTYPE = "lendingType"
    REGION = "region"

FILENAME = "filename"

aggregate = {
    Aggregation.COUNTRY: {
        FILENAME: "country.json"
    },
    Aggregation.INCOMELEVEL: {
        FILENAME: "incomeLevel.json"
    },
    Aggregation.LENDINGTYPE: {
        FILENAME: "lendingType.json"
    },
    Aggregation.REGION: {
        FILENAME: "region.json"
    },
}