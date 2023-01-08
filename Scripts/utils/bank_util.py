# Inspired by 
# https://github.com/kresusapp/kresus/blob/f6e2a03a5c2e3f936b4500be66e3dfd3123b941b/server/providers/woob/py/main.py#L493
# https://github.com/KDE/kmymoney/blob/65165330f65453e7f4fdb9604a6f5bbc8cfd2f31/kmymoney/plugins/woob/interface/kmymoneywoob.py#L78

import os

from woob.core import Woob
from pprint import pprint
import jsons


modulename=os.getenv('BANK')

id_parameters={
    "login": os.getenv('BANK_USER'),
    "password": os.getenv('BANK_PW')
}
session = dict()
account_id = os.getenv('BANK_ACCOUNT_ID')

woob = Woob()
repositories = woob.repositories
minfo = repositories.get_module_info(modulename)

backend = woob.build_backend(
    modulename,
    id_parameters,
    session
)

## To get accounts ids:
# accounts = backend.iter_accounts()
# results = {}
# for account in accounts:
#     print(account)
#     results[account.id] = {'name': account.label,
#                             'balance': float(account.balance),
#                             'type': account.type,
#                             }
#     print("--------------------------------------")
# pprint(results)

account = backend.get_account(account_id)
result = {
    'AccountId': account.id,
    'label': account.label,
    'balance': float(account.balance),
    'type': account.type,
}
print(result)

## For exhaustive information:
# res = jsons.dump(account)
# pprint(res)