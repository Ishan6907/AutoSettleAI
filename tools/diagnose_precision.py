import json

d = json.load(open('runs/photo_eval.json'))
print('Total photos:', d['summary']['photos'])
print('Found:', d['summary']['found'])
print('Missed:', d['summary']['missed'])
print('Extra (False Positives):', d['summary']['extra'])
print('Precision:', d['summary']['part_precision'])
print('Recall:', d['summary']['part_recall'])
print('=' * 60)

for r in d['rows']:
    if not r['pred']:
        continue
    truth_parts = {t['part']: t for t in r['truth']}
    pred_parts = {p['part_class']: p for p in r['pred']}
    extras = [p for p in pred_parts if p not in truth_parts]
    matches = [p for p in pred_parts if p in truth_parts]
    print(f"File: {r['file']}")
    print(f"  Truth: {[(t['part'], t['damage'], t['severity']) for t in r['truth']]}")
    print(f"  Pred:  {[(p['part_class'], p['damage_class'], p['severity'], p.get('damage_conf'), p.get('damage_source')) for p in r['pred']]}")
    if extras:
        print(f"  -> FALSE POSITIVES (Extra): {extras}")
    if matches:
        print(f"  -> TRUE POSITIVES (Matched): {matches}")
    print("-" * 50)
