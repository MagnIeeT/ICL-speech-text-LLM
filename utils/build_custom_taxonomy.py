#!/usr/bin/env python3
"""Add custom-cluster label columns to an SLU dataset for the "LLM clustering into
user-defined clusters" eval. One clustering (intent->cluster), THREE label styles:
  cat_meaningful | cat_neutral | cat_symbol   (same ground truth, different names)
Definitions live in the config prompt. Usage:
  python utils/build_custom_taxonomy.py --dataset minds14 --src <in> --out <out>
"""
import argparse
from collections import Counter
from datasets import load_from_disk

# intent -> canonical cluster (c1 info / c2 action / c3 problem)
MAPS = {
 "minds14": {
   "balance":"c1","latest_transactions":"c1","atm_limit":"c1","address":"c1","abroad":"c1",
   "pay_bill":"c2","direct_debit":"c2","cash_deposit":"c2","high_value_payment":"c2",
   "app_error":"c3","card_issues":"c3","freeze":"c3","business_loan":"c3","joint_account":"c3",
 },
 "skit": {
   "branch_address":"c1","past_transactions":"c1","outstanding_balance":"c1","ifsc_code":"c1","balance_enquiry":"c1","dispatch_status":"c1",
   "activate_card":"c2","generate_pin":"c2","change_limit":"c2","loan_query":"c2",
   "card_issue":"c3","unauthorised_transaction":"c3","block":"c3","lost":"c3",
 },
}
STYLE = {
 "cat_meaningful": {"c1":"info_lookup","c2":"account_action","c3":"problem_report"},
 "cat_neutral":    {"c1":"category_1","c2":"category_2","c3":"category_3"},
 "cat_symbol":     {"c1":"vfeld","c2":"qomr","c3":"tirsk"},
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,choices=list(MAPS))
    p.add_argument("--src",required=True); p.add_argument("--out",required=True)
    p.add_argument("--intent_key",default="intent_label")
    a=p.parse_args()
    m=MAPS[a.dataset]; ds=load_from_disk(a.src)
    missing=set(ds[a.intent_key])-set(m); assert not missing, f"unmapped intents: {missing}"
    def add(r):
        c=m[r[a.intent_key]]; return {col:STYLE[col][c] for col in STYLE}
    ds=ds.map(add)
    print("cluster dist:",dict(sorted(Counter(m[i] for i in ds[a.intent_key]).items())))
    ds.save_to_disk(a.out); print(f"saved {len(ds)} -> {a.out}")

if __name__=="__main__": main()
