# Hiver AI Support Agent — Failure Analysis

This report summarizes observed failure modes from the frozen 200-example evaluation and the 30-example human/judge reply-quality validation sample.

## 1. Evaluation Snapshot

- Full-agent intent accuracy: 10.00%
- Full-agent escalation accuracy: 54.50%
- Self-reported grounded-output rate: 97.00%
- Runtime errors: 3

### Retrieval
- Lexical Recall@1: 0.08
- Lexical Recall@3: 0.135
- Lexical Recall@5: 0.145
- Intent-aware Recall@1: 0.09
- Intent-aware Recall@3: 0.135
- Intent-aware Recall@5: 0.15

## 2. Top Failure Modes

### Failure Mode 1: PURCHASE_REPAIR_WARRANTY → OTHER_UNCLEAR

Observed 17 example(s) with gold intent `PURCHASE_REPAIR_WARRANTY` predicted as `OTHER_UNCLEAR`.

**Examples:**
- `1677876` — @AppleSupport @115858 can we get an update on bug fixes and emojis fixes? This A? Is mad annoying 🙄
- `1995399` — first of all @115858 the letter “i” is acting funny. Please fix tysm

**Hypothesis:** The classifier is confusing overlapping intent boundaries or relying on a symptom that is not the customer's primary support need.

### Failure Mode 2: ACCOUNT_SECURITY → OTHER_UNCLEAR

Observed 16 example(s) with gold intent `ACCOUNT_SECURITY` predicted as `OTHER_UNCLEAR`.

**Examples:**
- `2005374` — @115858 The worst part is that it looks like you’re typing an I️ until you hit send and then see you actually sent an I️
- `2923598` — Dear @AppleSupport, #iOS11 has completely hosed #CarPlay in my BMW. It continually disconnects, reconnects. Extreme lag &amp; freezing on phone &amp; HUD. Locked up phone to point I had to hard reset. It was fine before moving to iOS11. Unp

**Hypothesis:** The classifier is confusing overlapping intent boundaries or relying on a symptom that is not the customer's primary support need.

### Failure Mode 3: CONNECTIVITY → OTHER_UNCLEAR

Observed 15 example(s) with gold intent `CONNECTIVITY` predicted as `OTHER_UNCLEAR`.

**Examples:**
- `1745486` — Honestly though can someone help me figure out what this @115858 problem is?? I don't get it
- `1099654` — @AppleSupport Yes, it went from not connecting at all to only playing Apple Music (rather than Spotify of podcasts)

**Hypothesis:** The classifier is confusing overlapping intent boundaries or relying on a symptom that is not the customer's primary support need.

### Failure Mode 4: APPS_MEDIA → OTHER_UNCLEAR

Observed 15 example(s) with gold intent `APPS_MEDIA` predicted as `OTHER_UNCLEAR`.

**Examples:**
- `1897912` — Man @115858 is tripping. Just bc a new phone comes out they think they slick making my phone mess up.
- `457198` — Installed #highsierra and my late 2011 Mac is not working, it had no issues previously @AppleSupport. I got it booted up, after reinstalling High Sierra, then I shut down &amp; it does not work. Here’s what my screen looked like the other d

**Hypothesis:** The classifier is confusing overlapping intent boundaries or relying on a symptom that is not the customer's primary support need.

### Failure Mode 5: DEVICE_HARDWARE → OTHER_UNCLEAR

Observed 15 example(s) with gold intent `DEVICE_HARDWARE` predicted as `OTHER_UNCLEAR`.

**Examples:**
- `1697477` — @115858 you guys need to fix this glitch before exclamation point and question mark box get a galaxy
- `706147` — @115858 @AppleSupport Just did your update. All my photos and videos are gone. How do I recover them? Or how do I reverse the update?

**Hypothesis:** The classifier is confusing overlapping intent boundaries or relying on a symptom that is not the customer's primary support need.

## 3. Escalation Failure Modes

- Gold `False` → Predicted `True`: 62 example(s)
- Gold `True` → Predicted `False`: 29 example(s)

## 4. High-Confidence Intent Errors

- `388905`: IOS_UPDATE → BATTERY_POWER; confidence=0.95
  - @208024 @115858 I keep having to charge 2-3 times a day and my phone is 6 months old. Wtf is up with iOS 11
- `659463`: PERFORMANCE_STABILITY → BATTERY_POWER; confidence=0.95
  - @115858  you suck! With the latest iOS, battery drains at least 40% faster. #apple
- `2040306`: FEATURE_HOW_TO → DEVICE_HARDWARE; confidence=0.95
  - Er WTF happened to my screen, @AppleSupport?!
@603285 https://t.co/UN7JYVAaxu
- `2355025`: OTHER_UNCLEAR → IOS_UPDATE; confidence=0.95
  - Trying To Update My IPhone And This Keeps Popping Up Even Though I Am Connected To The Internet! Helppppp. @AppleSupport @115858 https://t.co/iJy2qWTXHD
- `2695019`: DEVICE_HARDWARE → FEATURE_HOW_TO; confidence=0.95
  - Recently updated OS. Now I get harassment screen "What's new in Photos" every time I open the damn thing. How do I turn it off, forever? @AppleSupport
- `794627`: APPS_MEDIA → BATTERY_POWER; confidence=0.95
  - Since the most recent update my iPhone 6 won't hold a charge nearly as long as it used to. Why @115858 ??
- `243287`: OTHER_UNCLEAR → APPS_MEDIA; confidence=0.95
  - @AppleSupport is the App Store down? I can’t download or update any apps, the circle keeps spinning and I’ve been trying for over 12 hours
- `1983018`: ACCOUNT_SECURITY → CONNECTIVITY; confidence=0.95
  - @AppleSupport My iPhone 5S network has been showing ""No service"" since 3 days. Network guys @117128 have given up. Please help :(
- `2639698`: OTHER_UNCLEAR → IOS_UPDATE; confidence=0.95
  - @AppleSupport my iMac has been stuck on loading screen since #HighSierra update, anyone have a fix? ✌🏾 https://t.co/3kmBceX4Ym
- `2255263`: PERFORMANCE_STABILITY → ACCOUNT_SECURITY; confidence=0.95
  - @AppleSupport Hi someone hacked my Apple ID and changed it, so the one I created is not recognized. Is there something that can be done so I don't lose all of my music?

## 5. Reply-Quality Judge Limitations

The LLM judge was evaluated against 30 human-rated responses. Judge-human agreement was weak, so judge scores are treated as diagnostic rather than ground truth.

- Human review sample: 30
- Human overall mean: 4.23
- Judge overall mean: 1.60
- Judge unsupported-claim rate: 0.00%

## 6. Operational Issues

- `2172859`: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01kyp9yqzhfv399y6gfaz71v9x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198504, Requested 1661. Please try again in 1m11.28s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
- `1551810`: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01kyp9yqzhfv399y6gfaz71v9x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199010, Requested 1236. Please try again in 1m46.271999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
- `2048961`: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01kyp9yqzhfv399y6gfaz71v9x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199851, Requested 1262. Please try again in 8m0.816s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}

## 7. What We Would Improve With One More Week

1. Replace the lexical retrieval layer with a semantic embedding retriever and evaluate Recall@K again.
2. Expand human evaluation beyond 200 intent examples and stratify by ambiguous intent boundaries.
3. Calibrate classifier confidence using a held-out calibration set instead of raw model confidence.
4. Improve primary-intent precedence for multi-symptom Twitter messages.
5. Add stronger response-grounding checks before a reply is exposed to a customer.