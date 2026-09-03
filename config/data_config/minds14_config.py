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

# Definition-sensitivity ablation: same task/labels, only the wording after each "- label:" changes.
_PROMPT_PARA = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- abroad: the customer asks about using their bank account while traveling or living overseas
- address: the customer wants to update or check the postal address on their account
- app_error: the customer reports something going wrong in the mobile banking app
- atm_limit: the customer asks how much cash they can withdraw from an ATM
- balance: the customer wants to know how much money is in their account
- business_loan: the customer asks about borrowing money for a business
- card_issues: the customer reports a problem with their debit or credit card
- cash_deposit: the customer asks about paying physical cash into their account
- direct_debit: the customer asks about arranging or changing a recurring direct debit
- freeze: the customer wants to suspend or block their account or card
- high_value_payment: the customer asks about sending a large sum of money
- joint_account: the customer asks about opening or managing an account shared with someone else
- latest_transactions: the customer asks about their most recent account activity
- pay_bill: the customer wants to settle a bill

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

_PROMPT_TERSE = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- abroad: banking while overseas
- address: change or check account address
- app_error: banking app problem
- atm_limit: ATM withdrawal limit
- balance: account balance
- business_loan: loan for a business
- card_issues: debit/credit card problem
- cash_deposit: deposit cash
- direct_debit: manage a direct debit
- freeze: block account or card
- high_value_payment: large payment
- joint_account: joint account query
- latest_transactions: recent transactions
- pay_bill: pay a bill

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

_PROMPT_VERBOSE = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- abroad: the customer is making an enquiry related to accessing or using their banking services while they are abroad, travelling internationally, or living in another country
- address: the customer is requesting to change, update, or ask a question about the residential or postal address registered on their bank account
- app_error: the customer is reporting a technical error, bug, crash, or other malfunction encountered while using the bank's mobile or online application
- atm_limit: the customer is enquiring about the maximum amount of cash they are permitted to withdraw from an automated teller machine
- balance: the customer is enquiring about the current available balance or amount of money held in their bank account
- business_loan: the customer is making an enquiry about applying for or managing a loan intended for business or commercial purposes
- card_issues: the customer is reporting a fault or difficulty they are experiencing with their debit card or credit card
- cash_deposit: the customer is enquiring about the process of depositing or paying physical cash into their bank account
- direct_debit: the customer is enquiring about setting up, amending, cancelling, or otherwise managing a recurring direct debit arrangement
- freeze: the customer is requesting to temporarily freeze, suspend, or block access to their account or payment card
- high_value_payment: the customer is enquiring about making a large or high-value monetary payment or transfer
- joint_account: the customer is enquiring about opening, closing, or managing a bank account shared jointly with another person
- latest_transactions: the customer is enquiring about their most recent transactions and account activity
- pay_bill: the customer is requesting to make a payment towards a bill

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
        prompt_variants={"paraphrase": _PROMPT_PARA, "terse": _PROMPT_TERSE, "verbose": _PROMPT_VERBOSE},
        valid_labels=list(_LABELS),
        completion_key="intent_label",
        text_key="text",
        max_new_tokens=16,
    )


MINDS14_EN_CONFIG = _cfg(DatasetType.MINDS14_EN, "MINDS14_EN")
MINDS14_FR_CONFIG = _cfg(DatasetType.MINDS14_FR, "MINDS14_FR")
MINDS14_KO_CONFIG = _cfg(DatasetType.MINDS14_KO, "MINDS14_KO")  # Korean: distant, AF3-unseen SLU language
