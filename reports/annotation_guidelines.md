# AppleSupport Golden-Set Annotation Guidelines

## Goal

Assign the customer's PRIMARY support need to exactly one intent.

The task is not to describe every symptom in the message. Choose the
single intent that best represents the main problem the customer wants
AppleSupport to solve.

## Intent precedence rules

### 1. ACCOUNT_SECURITY

Use for:
- Apple ID access
- Password reset/lockout
- Account verification
- Authentication
- Account security

Account-specific access problems take precedence over general symptoms.

### 2. PURCHASE_REPAIR_WARRANTY

Use for:
- Orders
- Refunds
- Purchases
- Repairs
- Warranty
- Replacement
- Service appointments

A request for repair/service takes precedence over the underlying
device symptom when the customer is explicitly asking for repair/service.

### 3. CONNECTIVITY

Use for:
- Wi-Fi
- Bluetooth
- Cellular/mobile network
- Hotspot
- Connection failures

If a customer says a device cannot connect, prefer CONNECTIVITY even if
an update is mentioned as the suspected cause.

### 4. BATTERY_POWER

Use for:
- Battery drain
- Charging problems
- Battery capacity/life
- Power/charging behavior

If the primary symptom is battery drain or charging, use BATTERY_POWER
even when an iOS update is mentioned as the suspected cause.

### 5. DEVICE_HARDWARE

Use for:
- Physical damage
- Broken screen
- Broken keys
- Speaker/camera hardware
- Hardware failure
- Device not powering on

Do NOT use DEVICE_HARDWARE merely because the customer mentions a device,
keyboard, screen, or phone. There must be evidence of a physical/hardware
problem.

### 6. PERFORMANCE_STABILITY

Use for:
- Freezing
- Crashing
- Restarting
- Slowness
- General instability

Use this when instability itself is the primary issue.

### 7. APPS_MEDIA

Use for:
- Apple Music
- Photos
- Messages
- FaceTime
- iMovie
- Media
- App-specific functionality or failures

App/service availability or failures belong here unless the primary
problem is clearly CONNECTIVITY.

### 8. FEATURE_HOW_TO

Use when the customer primarily asks:
- How do I...?
- Where is...?
- How can I enable/disable...?
- How does this feature work?

A how-to request takes precedence over the underlying application or
feature name.

### 9. IOS_UPDATE

Use for:
- Failed update installation
- Update download problems
- Update verification
- Update errors
- Update availability

If the update itself is the problem, use IOS_UPDATE.

If the update is merely mentioned as the suspected cause of another
primary symptom (for example battery drain or Bluetooth failure), label
the primary symptom instead.

### 10. OTHER_UNCLEAR

Use when:
- The message is too vague
- The problem cannot be identified reliably
- Multiple unrelated problems are present and no primary issue is clear
- The message contains insufficient information

## Multi-issue rule

When multiple symptoms occur, choose the customer's PRIMARY request.

Example:

"After iOS 11 my battery drains and Bluetooth stopped working."

If the customer is primarily complaining about Bluetooth connection:
→ CONNECTIVITY

If primarily complaining about battery drain:
→ BATTERY_POWER

If no primary issue is identifiable:
→ OTHER_UNCLEAR

## Evidence rule

Do not infer hardware failure from emotional language.

Do not infer account problems unless account access/security is involved.

Do not infer an intent merely from a keyword.

## Escalation labeling

TRUE:
- Account-specific investigation required
- Purchase/order-specific investigation required
- Repair/service/hardware intervention likely required
- Sensitive or high-risk case
- Insufficient information where safe automated handling is unreliable

FALSE:
- A general troubleshooting or informational response can reasonably
  address the request without account-specific intervention.