from utils.environment import get_env_path
from .master_config import DatasetType, DatasetSplit, DatasetConfig

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
