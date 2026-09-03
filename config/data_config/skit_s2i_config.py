from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

# Definition-sensitivity ablation: same task/labels, only the wording after each "- intent:" changes.
_PARA = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- branch_address: the customer asks where a bank branch is located
- activate_card: the customer asks about activating a card
- past_transactions: the customer asks about earlier transactions within a certain period
- dispatch_status: the customer asks whether their card has been dispatched
- outstanding_balance: the customer asks how much is still owed on a card
- card_issue: the customer reports trouble using a card
- ifsc_code: the customer asks for the IFSC code of a branch
- generate_pin: the customer asks about changing or creating a new card PIN
- unauthorised_transaction: the customer reports a transaction they did not authorise
- loan_query: the customer asks about the different loans on offer
- balance_enquiry: the customer asks about their account balance
- change_limit: the customer asks about changing a card's limit
- block: the customer asks about blocking a card or banking product
- lost: the customer reports that a card has been lost

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

_TERSE = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- branch_address: branch location
- activate_card: activate a card
- past_transactions: past transactions
- dispatch_status: card dispatch status
- outstanding_balance: outstanding card balance
- card_issue: card usage problem
- ifsc_code: branch IFSC code
- generate_pin: new card PIN
- unauthorised_transaction: fraudulent transaction
- loan_query: loan enquiry
- balance_enquiry: account balance
- change_limit: change card limit
- block: block a card
- lost: lost card

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

_VERBOSE = """You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- branch_address: the customer is enquiring about the physical location or address of a bank branch
- activate_card: the customer is enquiring about the process of activating a card or card product
- past_transactions: the customer is enquiring about their previous or historical transactions within a specific time period
- dispatch_status: the customer is enquiring about the current dispatch or delivery status of a card product they are expecting
- outstanding_balance: the customer is enquiring about the outstanding amount or balance owed on a card product
- card_issue: the customer is reporting a problem or difficulty they are experiencing while using a card product
- ifsc_code: the customer is enquiring about the IFSC code of a particular bank branch
- generate_pin: the customer is enquiring about changing, resetting, or generating a new PIN for a card product
- unauthorised_transaction: the customer is reporting a transaction that was unauthorised, fraudulent, or not made by them
- loan_query: the customer is enquiring about the various kinds of loans that are available
- balance_enquiry: the customer is enquiring about the balance held in their bank account
- change_limit: the customer is enquiring about changing or adjusting the spending or usage limit on a card product
- block: the customer is enquiring about blocking or deactivating a card or banking product
- lost: the customer is reporting that they have lost a card product

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only"""

SKIT_S2I_CONFIG = DatasetConfig(
    name=DatasetType.SKIT_S2I,
    paths={
        DatasetSplit.TRAIN: get_env_path("SKIT_S2I_TRAIN_PATH"),
        DatasetSplit.VAL: get_env_path("SKIT_S2I_VAL_PATH"),
        DatasetSplit.TEST: get_env_path("SKIT_S2I_TEST_PATH"),
    },
    prompt_template="""You are a banking voice-assistant intent classifier. Based on the audio, respond with EXACTLY ONE label from the following options:

Available intents:
- branch_address: enquiry about bank branch location
- activate_card: enquiry about activating card products
- past_transactions: enquiry about past transactions in a specific time period
- dispatch_status: enquiry about the dispatch status of card products
- outstanding_balance: enquiry about outstanding balance on card products
- card_issue: report about an issue with using card products
- ifsc_code: enquiry about IFSC code of bank branch
- generate_pin: enquiry about changing or generating a new pin for a card product
- unauthorised_transaction: report about an unauthorised or fraudulent transaction
- loan_query: enquiry about different kinds of loans
- balance_enquiry: enquiry about bank account balance
- change_limit: enquiry about changing the limit for card products
- block: enquiry about blocking a card or banking product
- lost: report about losing a card product

Guidelines:
- Choose exactly one intent from the list above
- Respond with the label only""",
    prompt_variants={"paraphrase": _PARA, "terse": _TERSE, "verbose": _VERBOSE},
    valid_labels=[
        "branch_address",
        "activate_card",
        "past_transactions",
        "dispatch_status",
        "outstanding_balance",
        "card_issue",
        "ifsc_code",
        "generate_pin",
        "unauthorised_transaction",
        "loan_query",
        "balance_enquiry",
        "change_limit",
        "block",
        "lost",
    ],
    completion_key="intent_label",
    text_key="text",
    max_new_tokens=16,
    # US/UK spelling alias so "unauthorized_transaction" isn't marked invalid.
    label_mapping={"unauthorized_transaction": "unauthorised_transaction"},
)
