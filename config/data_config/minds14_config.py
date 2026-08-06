from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# MInDS-14 banking intent (definable classes → works with zero-shot legend).
# Two variants: English (en-US+en-GB+en-AU) and French (fr-FR) for a cross-lingual probe.
# Same English legend + definitions; only the audio language differs.
_PROMPT = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- abroad: enquiry about banking while abroad or traveling overseas
- address: request to change or ask about the address on the account
- app_error: report of an error or problem in the banking app
- atm_limit: enquiry about the ATM cash withdrawal limit
- balance: enquiry about the account balance
- business_loan: enquiry about a loan for a business
- card_issues: report of a problem with a debit or credit card
- cash_deposit: enquiry about depositing cash into the account
- direct_debit: enquiry about setting up or managing a direct debit
- freeze: request to freeze or block the account or card
- high_value_payment: enquiry about making a large / high-value payment
- joint_account: enquiry about opening or managing a joint account
- latest_transactions: enquiry about recent or latest transactions
- pay_bill: request to pay a bill

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

_LABELS = ['abroad','address','app_error','atm_limit','balance','business_loan','card_issues',
           'cash_deposit','direct_debit','freeze','high_value_payment','joint_account',
           'latest_transactions','pay_bill']


def _cfg(name, prefix):
    return DatasetConfig(
        name=name,
        paths={
            DatasetSplit.TRAIN: get_env_path(f"{prefix}_TRAIN_PATH"),
            DatasetSplit.VAL: get_env_path(f"{prefix}_VAL_PATH"),
            DatasetSplit.TEST: get_env_path(f"{prefix}_TEST_PATH"),
        },
        prompt_template=_PROMPT,
        valid_labels=list(_LABELS),
        completion_key="intent_label",
        text_key="text",
        max_new_tokens=16,
    )


MINDS14_EN_CONFIG = _cfg(DatasetType.MINDS14_EN, "MINDS14_EN")
MINDS14_FR_CONFIG = _cfg(DatasetType.MINDS14_FR, "MINDS14_FR")
MINDS14_KO_CONFIG = _cfg(DatasetType.MINDS14_KO, "MINDS14_KO")  # Korean: distant, AF3-unseen SLU language
