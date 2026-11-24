---
name: hiauditagent-explainer
description: Code explanation and understanding for Solidity smart contracts. Triggers on hiauditagent explain, hiauditagent what does, or hiauditagent walkthrough.
---

# HiAuditAgent Explainer

Security-aware code explanation for Solidity smart contracts. Every explanation maps trust boundaries, highlights security-relevant patterns, and identifies potential risk surfaces - turning code comprehension into the first layer of defense.

**Note:** This skill explains code. For security analysis, use hiauditagent-analyzer.

## When to Activate

- User wants to understand code
- User asks what something does
- User wants a walkthrough
- User asks about a contract's architecture or design

## When NOT to Use

- User wants security analysis (→ hiauditagent-analyzer)
- User wants to find vulnerabilities (→ hiauditagent-analyzer)
- User wants to validate an attack (→ hiauditagent-validate)
- User wants gas optimization (→ hiauditagent-gas)

## Process

1. **Context** - Check README if explaining core contract (understand project purpose and trust model)
2. **Read** - Get the code and surrounding context (inheritance, imports, related functions)
   - **For a specific function:** Run **hiauditagent_scan** if a report is not already loaded, then read the function and its callers/callees in source to explain data flow.
3. **Explain** - Follow the structured explanation framework below.
4. **Offer** - "Want me to explain more?" or "Check for vulnerabilities?"

## Adapt to Audience

Infer from how user asks, or ask if unclear:
- "what is a modifier?" → **Beginner** (use analogies, define jargon, explain EVM basics)
- "walk me through this" → **Experienced** (patterns, trade-offs, edge cases, gas implications)
- "what are the trust assumptions?" → **Auditor** (attack surface, state changes, invariants, formal properties)

Default to experienced if unclear.

## Explanation Framework

### Layer 1: Purpose and Context
```
What does this contract/function DO?
- One-paragraph summary of its role in the protocol
- What problem it solves
- How it fits into the broader system architecture
```

### Layer 2: Execution Flow
```
Step-by-step walkthrough:
- Lines X-Y: [what this section does]
- Line Z: [what this does and WHY]
- Focus on state changes, external calls, and branching logic
```

### Layer 3: Security Architecture
```
How is this code PROTECTED?
- Access control: Who can call this? What modifiers restrict access?
- Input validation: What checks guard against bad input?
- State management: What invariants does this function maintain?
- External interactions: What contracts does this call? Are return values checked?
- Reentrancy: Does it follow CEI pattern? Are there callback risks?
```

### Layer 4: Trust Boundaries
```
What TRUST ASSUMPTIONS does this code make?
- Caller trust: Does it assume msg.sender is honest?
- External contract trust: Does it assume return values are correct?
- Oracle trust: Does it assume price data is fresh and accurate?
- Admin trust: Does it assume the owner acts in good faith?
- Timing trust: Does it assume block.timestamp is reliable?
```

### Layer 5: Data Flow and Invariants
```
What MUST always be true?
- Accounting invariants (balances, supplies, shares)
- Authorization invariants (role hierarchy)
- Temporal invariants (ordering, deadlines)
- Conservation invariants (value in == value out + fees)
```

## Output Structure

```
## [Contract/Function Name]

**Overview:** One paragraph - what does this do and why does it exist?

**Execution Flow:**
- Lines X-Y: [what this section does]
- Line Z: [what this does]

**Design Patterns:**
- [CEI, Pull-over-push, Checks-Effects-Interactions, etc.]
- [Why each pattern was chosen]

**Security Architecture:**
- Access control: [who can call, what restrictions]
- Input validation: [what's checked, what's NOT checked]
- External interactions: [what's called, return value handling]

**Trust Boundaries:**
- [Who/what is trusted and why]
- [What happens if trust is violated]

**Invariants:**
- [What must always hold true for this code to be correct]

**Risks Noted:** (if any spotted during explanation)
- [Not a full audit, but flag obvious concerns]
```

## Understanding a Function

When the user asks about a **specific function** (e.g. "explain withdraw", "what does claim do?"):
1. Run **hiauditagent_scan** if a report is not already loaded.
2. Read the function and its callers/callees in source (user-defined code only).
3. Describe the function's place in the call graph so the user gets a clear data-flow picture.

## Security-Relevant Patterns to Always Identify

When explaining code, always flag which of these patterns are present (or absent):

| Pattern | What to Note |
|---------|-------------|
| **CEI (Checks-Effects-Interactions)** | Is it followed? Any deviations? |
| **Pull-over-Push** | Are users pulling their funds vs protocol pushing? |
| **Reentrancy Guards** | Present? Which functions? Cross-function coverage? |
| **Access Control Model** | Ownable? Role-based? Multi-sig? Timelock? |
| **Pausability** | Can the protocol be paused? Who can pause? |
| **Upgrade Pattern** | Transparent? UUPS? Diamond? Storage gaps? |
| **Oracle Integration** | Which oracle? Staleness checks? Fallback? |
| **Token Handling** | SafeERC20? Fee-on-transfer aware? Rebasing aware? |

## Explaining Inheritance and Dependencies

When a contract inherits from others:
1. Map the full inheritance chain (top to bottom)
2. Identify which functions are overridden and what behavior changes
3. Flag shadow variables or function selector collisions
4. Note storage layout implications (especially for upgradeable contracts)
5. List external dependencies and their trust implications

```
Inheritance: Vault → ERC4626 → ERC20 → IERC20
                  → Ownable2Step
                  → ReentrancyGuard
                  → Pausable

External deps: OpenZeppelin v4.9.3 (pinned)
               Chainlink AggregatorV3Interface
```

## Explaining Value Flow

For any function that handles ETH or tokens:

```
Value Flow for deposit():
  IN:  User sends ETH via msg.value
  ↓    Validation: amount > minDeposit, contract not paused
  ↓    Accounting: shares minted = amount * totalShares / totalAssets
  ↓    State update: balances[user] += shares, totalShares += shares
  OUT: Shares credited to user (no outbound transfer)

  Edge cases:
  - First deposit (totalShares == 0): fixed share ratio used
  - Rounding: favors protocol (rounds DOWN on shares minted)
```

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "This is standard ERC20, I'll skip details" | User asked for explanation. Explain it. Standard code can have non-standard modifications. |
| "The function name is self-explanatory" | Names can be misleading. Read the actual implementation. |
| "I'll just give a quick summary" | If user wanted a summary, they'd read the docs. Provide layered depth. |
| "This library is well-known" | Explain how THIS code uses the library. Integration is where bugs hide. |
| "The comments explain it" | Comments can be outdated or wrong. Explain the actual code, not comments. |
| "The security patterns aren't relevant to explanation" | Security context IS the explanation. Every function has a trust model. |
| "I don't need to check the call graph" | Without call graph context, you'll miss cross-function dependencies and data flow. |
| "The modifiers are probably standard" | Always verify modifier implementations. Custom modifiers are common and can have bugs. |
