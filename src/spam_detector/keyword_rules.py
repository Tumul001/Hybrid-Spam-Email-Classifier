"""
keyword_rules.py — Hand-crafted spam keyword/signal rules.

BERT handles semantic understanding. These rules handle explicit,
predictable spam patterns across 10 categories.
Rules are compiled once at import time for maximum performance.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SpamRule:
    name: str
    category: str
    description: str
    weight: float
    pattern: re.Pattern


# (name, category, description, weight, regex)
_RAW: list[tuple[str, str, str, float, str]] = [

    # ── MARKETING ────────────────────────────────────────────────────────────
    ("unsubscribe",         "marketing", "Unsubscribe link (CAN-SPAM legally required)",             0.55, r"\bunsubscribe\b"),
    ("manage_preferences",  "marketing", "Manage preferences/subscription footer",                  0.50, r"\bmanage\s+(preferences|settings|subscription)\b"),
    ("view_in_browser",     "marketing", "View in browser link",                                    0.45, r"\bview\s+in\s+(browser|your\s+browser|web\s+browser)\b"),
    ("newsletter",          "marketing", "Self-identifies as a newsletter",                         0.45, r"\bnewsletter\b"),
    ("opt_out",             "marketing", "Opt-out option",                                          0.40, r"\bopt[- ]out\b"),
    ("weekly_digest",       "marketing", "Weekly/monthly digest label",                             0.40, r"\b(weekly|monthly|daily)\s+digest\b"),
    ("receiving_because",   "marketing", "Explains why recipient is on the list",                   0.40, r"\byou\s+(are\s+)?receiving\s+this\s+(email|message|newsletter)\b"),
    ("forwarding_cta",      "marketing", "Asks to forward the email",                               0.30, r"\bforward\s+this\s+(email|message|newsletter)\b"),
    ("click_subscribe",     "marketing", "Asks to click to subscribe",                              0.30, r"\bclick\s+here\s+to\s+subscribe\b|\bsubscribe\s+to\s+(our|this|the)\b"),
    ("company_address",     "marketing", "Physical mailing address (CAN-SPAM)",                     0.35, r"\b\d{2,5}\s+\w[\w\s]{2,30}(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Pl|Ct|Sq|Pkwy)\b"),
    ("promotional_ps",      "marketing", "P.S. / P.P.S. marketing tactic",                         0.25, r"\bP\.?P?\.?S\.?\b"),
    ("exclusive_offer",     "marketing", "Exclusive or special offer language",                     0.35, r"\b(exclusive|special|personaliz[se]d)\s+offer\b"),
    ("free_trial",          "marketing", "Free trial promotion",                                    0.35, r"\bfree\s+trial\b"),
    ("no_strings",          "marketing", "No strings attached / no commitment",                     0.35, r"\bno\s+strings\s+attached\b|\bno\s+commitment\b"),
    ("money_back",          "marketing", "Money-back guarantee",                                    0.30, r"\bmoney[- ]back\s+guarantee\b|\bsatisfaction\s+guaranteed\b"),
    ("cancel_anytime",      "marketing", "Cancel anytime (subscription upsell)",                    0.25, r"\bcancel\s+anytime\b|\bno\s+cancellation\s+fee\b"),
    ("limited_slots",       "marketing", "Limited slots / seats available",                         0.40, r"\b(limited\s+slots?|limited\s+seats?|spots?\s+(are\s+)?filling\s+fast)\b"),
    ("flash_sale",          "marketing", "Flash sale / clearance",                                  0.35, r"\b(flash\s+sale|clearance\s+sale|blowout\s+sale)\b"),
    ("promo_code",          "marketing", "Promotional / discount code",                             0.30, r"\b(promo\s+code|discount\s+code|coupon\s+code)\b"),
    ("free_shipping",       "marketing", "Free shipping offer",                                     0.25, r"\bfree\s+shipping\b"),
    ("bogo",                "marketing", "Buy one get one offer",                                   0.30, r"\b(buy\s+one\s+get\s+one|bogo)\b"),
    ("as_seen_on_tv",       "marketing", "As seen on TV",                                           0.40, r"\bas\s+seen\s+on\s+(tv|television)\b"),
    ("percent_off",         "marketing", "Percentage discount offer",                               0.25, r"\b\d{2,3}\s*%\s+off\b"),

    # ── PHISHING ─────────────────────────────────────────────────────────────
    ("verify_account",      "phishing",  "Asks to verify account",                                  0.75, r"\bverify\s+(your\s+)?(account|identity|email|information|details)\b"),
    ("account_suspended",   "phishing",  "Claims account is suspended/locked",                      0.80, r"\b(account|access)\s+(has\s+been\s+)?(suspended|locked|compromised|limited|restricted|disabled)\b"),
    ("unusual_activity",    "phishing",  "Claims unusual activity detected",                        0.75, r"\bunusual\s+(activity|sign[- ]in|login|access)\b"),
    ("confirm_identity",    "phishing",  "Asks to confirm identity",                                0.70, r"\bconfirm\s+(your\s+)?(identity|account|information|details)\b"),
    ("password_expired",    "phishing",  "Claims password has expired",                             0.80, r"\bpassword\s+(has\s+)?(expired|will\s+expire|expiring)\b"),
    ("update_payment",      "phishing",  "Asks to update payment information",                      0.75, r"\bupdate\s+(your\s+)?(payment|billing|credit\s+card|bank)\s+(information|info|details|method)\b"),
    ("action_required",     "phishing",  "Demands immediate action",                                0.60, r"\b(action\s+required|immediate\s+action|urgent\s+action)\b"),
    ("click_to_confirm",    "phishing",  "Asks to click to confirm/verify",                         0.65, r"\bclick\s+(here\s+to\s+)?(confirm|verify|validate|activate|reactivate)\b"),
    ("security_alert",      "phishing",  "Fake security alert",                                     0.65, r"\b(security\s+alert|security\s+warning|security\s+notice)\b"),
    ("login_attempt",       "phishing",  "Claims suspicious login attempt",                         0.70, r"\b(suspicious|unauthorized|unrecognized)\s+(login|sign[- ]in|access)\s*(attempt|detected)?\b"),
    ("dear_customer",       "phishing",  "Impersonal salutation (phishing tactic)",                 0.45, r"\bdear\s+(customer|user|member|account\s+holder|valued\s+client)\b"),
    ("billing_problem",     "phishing",  "Billing or payment problem claimed",                      0.70, r"\b(billing|payment)\s+(problem|issue|error|failed|declined|unsuccessful)\b"),
    ("reset_password",      "phishing",  "Asks to reset password (out of context)",                 0.55, r"\b(reset|change)\s+your\s+password\b"),
    ("account_flagged",     "phishing",  "Claims account has been flagged",                         0.70, r"\b(account|profile)\s+(has\s+been\s+)?(flagged|marked|reported)\b"),
    ("suspicious_transaction","phishing","Claims a suspicious transaction occurred",                 0.70, r"\bsuspicious\s+(transaction|charge|activity|purchase)\b"),
    ("otp_phishing",        "phishing",  "Asks for OTP or one-time password",                       0.65, r"\b(one[- ]time\s+password|otp|verification\s+code)\b.{0,50}\b(enter|provide|share|send)\b"),
    ("it_support",          "phishing",  "Fake IT/Microsoft/Apple support",                         0.70, r"\b(IT\s+support|Microsoft\s+support|Apple\s+support|help\s+desk)\b.{0,40}\b(contact|call|click)\b"),
    ("session_expired",     "phishing",  "Claims session has expired",                              0.60, r"\b(session|account)\s+has\s+(expired|timed\s+out)\b"),
    ("reenter_info",        "phishing",  "Asks to re-enter personal information",                   0.65, r"\bre[- ]?enter\s+(your\s+)?(password|details|information|credentials)\b"),

    # ── SCAM ─────────────────────────────────────────────────────────────────
    ("you_have_won",        "scam",      "Claims recipient has won something",                      0.85, r"\b(you\s+have\s+won|you\s+are\s+(a\s+)?winner|selected\s+as\s+the\s+winner)\b"),
    ("lottery_winner",      "scam",      "Lottery / sweepstakes winner",                            0.90, r"\b(lottery|sweepstakes|lotto|raffle)\s+(winner|prize|winnings)\b"),
    ("claim_prize",         "scam",      "Asks to claim a prize or reward",                         0.80, r"\bclaim\s+(your\s+)?(prize|reward|winnings|gift|package)\b"),
    ("million_dollars",     "scam",      "Implausible large sum of money",                          0.85, r"\b\d+[\.,]?\d*\s*(million|billion)\s*(dollars?|usd|gbp|euros?|pounds?)\b"),
    ("inheritance_funds",   "scam",      "Inheritance or fund transfer scam",                       0.90, r"\b(inheritance|unclaimed\s+funds|fund\s+transfer|estate\s+of)\b"),
    ("advance_fee",         "scam",      "Advance fee / processing fee scam",                       0.90, r"\b(advance\s+fee|processing\s+fee|transfer\s+fee|handling\s+fee)\b"),
    ("bank_details",        "scam",      "Requests bank account details",                           0.85, r"\bbank\s+account\s+details\b|\baccount\s+number\b.{0,30}\bprovide\b"),
    ("govt_official",       "scam",      "Impersonates government official",                        0.85, r"\bI\s+am\s+(a\s+)?(government\s+official|diplomat|minister|prince|king)\b"),
    ("confidential_deal",   "scam",      "Asks to keep deal confidential",                          0.75, r"\bkeep\s+(this\s+)?(confidential|secret|strictly\s+between\s+us)\b"),
    ("mystery_shopper",     "scam",      "Secret/mystery shopper scam",                             0.80, r"\b(secret|mystery)\s+shopper\b"),
    ("contact_urgently",    "scam",      "Asks to reply/contact urgently",                          0.55, r"\b(contact|reply\s+to)\s+(me\s+)?(urgently|immediately|as\s+soon\s+as\s+possible)\b"),
    ("western_union",       "scam",      "Asks to use wire transfer services",                      0.85, r"\b(western\s+union|money\s+gram|wire\s+transfer|swift\s+transfer)\b"),
    ("percentage_funds",    "scam",      "Offers percentage of funds (advance fee scam)",           0.85, r"\b\d{1,2}\s*%\s*(of\s+)?(the\s+)?(total\s+)?(funds|amount|sum|money)\b"),
    ("god_bless",           "scam",      "Religious appeal common in 419 scams",                    0.60, r"\b(god\s+bless|in\s+the\s+name\s+of\s+(god|allah|jesus))\b.{0,100}\b(money|funds|transfer|million)\b"),
    ("i_write_to_you",      "scam",      "Formal opening common in advance fee scams",              0.65, r"\bI\s+(write|am\s+writing)\s+to\s+you\b.{0,50}\b(funds|million|transfer|inheritance)\b"),

    # ── URGENCY ──────────────────────────────────────────────────────────────
    ("limited_time",        "urgency",   "Limited time offer",                                      0.45, r"\blimited[- ]time\s+(offer|deal|discount|sale|only)\b"),
    ("act_now",             "urgency",   "Act now / Don't delay",                                   0.40, r"\b(act\s+now|don.t\s+delay|don.t\s+miss\s+out|don.t\s+wait)\b"),
    ("expires_today",       "urgency",   "Offer expires today or soon",                             0.45, r"\b(expires?\s+today|expiring\s+soon|last\s+chance|ends?\s+tonight)\b"),
    ("today_only",          "urgency",   "Today only / 24 hours only",                              0.45, r"\b(today\s+only|24[- ]hours?\s+only|this\s+week\s+only)\b"),
    ("respond_immediately", "urgency",   "Demands immediate response",                              0.50, r"\b(respond|reply|contact\s+us)\s+(immediately|right\s+away|asap|urgently)\b"),
    ("time_running_out",    "urgency",   "Time is running out pressure",                            0.40, r"\b(time\s+is\s+running\s+out|before\s+it.s\s+too\s+late|don.t\s+let\s+this\s+pass)\b"),
    ("final_notice",        "urgency",   "Final / last notice language",                            0.55, r"\b(final\s+notice|last\s+notice|second\s+notice|last\s+warning)\b"),
    ("selling_out",         "urgency",   "Selling out fast / almost gone",                          0.40, r"\b(selling\s+out\s+fast|almost\s+gone|while\s+supplies\s+last)\b"),
    ("hours_left",          "urgency",   "Hours left / countdown",                                  0.40, r"\b\d+\s*hours?\s+(left|remaining|only)\b"),

    # ── FINANCIAL SPAM ────────────────────────────────────────────────────────
    ("make_money",          "financial", "Make money fast schemes",                                 0.70, r"\bmake\s+(money|cash)\s+(fast|quickly|online|from\s+home|at\s+home)\b"),
    ("work_from_home",      "financial", "Work from home spam",                                     0.55, r"\b(work\s+from\s+home|work\s+at\s+home)\s+(opportunity|job|business|income)\b"),
    ("earn_extra",          "financial", "Earn extra cash / passive income",                        0.60, r"\b(earn\s+(extra\s+)?(cash|money|income)|passive\s+income)\b"),
    ("guaranteed_returns",  "financial", "Guaranteed returns / risk-free investment",               0.75, r"\b(guaranteed\s+(return|income|profit|results?)|risk[- ]free\s+invest)\b"),
    ("no_experience",       "financial", "No experience needed (MLM marker)",                       0.55, r"\bno\s+(experience|skills?|investment|money\s+down)\s+(needed|required|necessary)\b"),
    ("double_money",        "financial", "Double / triple your money",                              0.80, r"\b(double|triple|10x)\s+(your\s+)?(money|investment|income|profit)\b"),
    ("debt_relief",         "financial", "Debt relief / credit repair scam",                        0.65, r"\b(debt\s+relief|credit\s+repair|debt\s+consolidation|get\s+out\s+of\s+debt)\b"),
    ("no_credit_check",     "financial", "No credit check (predatory lending)",                     0.60, r"\bno\s+credit\s+check\b|\bbad\s+credit\s+(ok|okay|welcome|accepted)\b"),
    ("binary_options",      "financial", "Binary options / forex trading spam",                     0.70, r"\b(binary\s+options|forex\s+trading\s+(signal|tip|alert|opportunity))\b"),
    ("stock_tips",          "financial", "Penny stock tips / insider trading spam",                 0.70, r"\b(penny\s+stocks?|stock\s+tip|hot\s+stock|insider\s+tip)\b"),
    ("financial_freedom",   "financial", "Financial freedom / independence pitch",                  0.50, r"\b(financial\s+freedom|financial\s+independence)\b.{0,60}\b(system|program|method|secret)\b"),

    # ── HEALTH / PHARMA ───────────────────────────────────────────────────────
    ("weight_loss",         "health",    "Weight loss product spam",                                0.65, r"\b(lose\s+weight\s+fast|weight\s+loss\s+(pill|supplement|secret|trick))\b"),
    ("miracle_cure",        "health",    "Miracle cure / treatment claims",                         0.70, r"\b(miracle\s+cure|miracle\s+treatment|miracle\s+supplement|wonder\s+drug)\b"),
    ("clinically_proven",   "health",    "False clinically proven claim",                           0.50, r"\b(clinically\s+proven|doctor\s+recommended|fda\s+approved)\b.{0,60}\b(buy|order|click|free)\b"),
    ("no_prescription",     "health",    "No prescription needed (pharma spam)",                    0.80, r"\b(no\s+prescription\s+(needed|required)|prescription\s+not\s+(needed|required))\b"),
    ("online_pharmacy",     "health",    "Online pharmacy spam",                                    0.75, r"\b(online\s+pharmacy|cheap\s+(medication|meds|pills)|discount\s+(pills|meds))\b"),
    ("enhancement_pills",   "health",    "Enhancement / enlargement product spam",                  0.80, r"\b(male\s+enhancement|enlargement\s+pill|erectile\s+dysfunction\s+pill)\b"),
    ("burn_fat",            "health",    "Fat burning / diet product spam",                         0.60, r"\b(burn\s+fat|fat\s+burner|metabolism\s+booster)\b.{0,40}\b(pill|supplement|product|buy)\b"),
    ("without_exercise",    "health",    "Weight loss without diet or exercise (false claim)",      0.70, r"\b(without\s+(diet|exercise|working\s+out)|no\s+(diet|exercise)\s+required)\b"),

    # ── TECH SPAM ─────────────────────────────────────────────────────────────
    ("virus_warning",       "tech",      "Fake virus / malware warning",                            0.75, r"\b(your\s+(computer|device|pc)\s+(has|is)\s+(a\s+)?(virus|infected|hacked|compromised))\b"),
    ("call_tech_support",   "tech",      "Fake tech support call prompt",                           0.80, r"\b(call\s+(microsoft|apple|google|windows)\s+(support|helpline))\b"),
    ("license_expired",     "tech",      "Software license expired (fake)",                         0.65, r"\b(windows|antivirus|software)\s+(license|subscription)\s+(has\s+)?(expired|expiring)\b"),
    ("suspicious_files",    "tech",      "Suspicious files detected (fake)",                        0.70, r"\b(suspicious\s+files?\s+detected|malware\s+detected|threats?\s+found)\b"),
    ("ip_hacked",           "tech",      "Your IP has been hacked (fake)",                          0.75, r"\byour\s+ip\s+(address\s+)?(has\s+been\s+)?(hacked|tracked|monitored|blacklisted)\b"),
    ("click_to_fix",        "tech",      "Click here to fix your computer",                         0.70, r"\bclick\s+(here\s+)?(to\s+)?(fix|scan|remove|protect)\s+(your\s+)?(computer|device|pc)\b"),
    ("remote_access",       "tech",      "Asks to install remote access software",                  0.80, r"\b(install|download|allow)\s+(remote\s+access|anydesk|teamviewer)\b"),

    # ── GAMBLING ─────────────────────────────────────────────────────────────
    ("online_casino",       "gambling",  "Online casino promotion",                                 0.70, r"\b(online\s+casino|casino\s+bonus|slot\s+machine)\b"),
    ("free_spins",          "gambling",  "Free spins / free bets (casino spam)",                    0.65, r"\b(free\s+spins?|free\s+bets?|no\s+deposit\s+bonus)\b"),
    ("sports_betting",      "gambling",  "Sports betting spam",                                     0.65, r"\b(sports?\s+betting|place\s+your\s+bet|bet\s+now|sure\s+bet|jackpot)\b"),
    ("winning_streak",      "gambling",  "Winning streak / guaranteed picks",                       0.70, r"\b(guaranteed\s+picks?|100%\s+winning|tipster)\b"),

    # ── CRYPTOCURRENCY ────────────────────────────────────────────────────────
    ("crypto_investment",   "crypto",    "Cryptocurrency investment spam",                          0.65, r"\b(crypto(currency)?\s+(investment|opportunity|profit|trading\s+bot))\b"),
    ("bitcoin_profit",      "crypto",    "Bitcoin / Ethereum profit claim",                         0.70, r"\b(bitcoin|ethereum|btc|eth)\s+(profit|earning|investment|opportunity)\b"),
    ("token_sale",          "crypto",    "ICO / token sale spam",                                   0.70, r"\b(token\s+sale|ico|initial\s+coin\s+offering|nft\s+(drop|mint|opportunity))\b"),
    ("defi_yield",          "crypto",    "DeFi yield farming spam",                                 0.65, r"\b(defi|yield\s+farming|staking\s+reward|liquidity\s+pool)\b.{0,60}\b(earn|profit|return)\b"),
    ("crypto_doubler",      "crypto",    "Crypto doubling scam",                                    0.90, r"\b(send\s+(bitcoin|btc|eth|crypto).{0,30}(receive|get\s+back)\s+double)\b"),

    # ── ADULT ─────────────────────────────────────────────────────────────────
    ("adult_dating",        "adult",     "Adult dating site spam",                                  0.80, r"\b(singles?\s+near\s+you|local\s+singles?|hot\s+singles?\s+near)\b"),
    ("explicit_invite",     "adult",     "Explicit content invitation",                             0.85, r"\b(see\s+my\s+(photos?|pics?|videos?)|watch\s+me\s+(live|online))\b"),
    ("adult_site",          "adult",     "Adult website promotion",                                 0.80, r"\b(adult\s+(website|content|dating)|xxx\s+(site|content))\b"),
    ("cam_site",            "adult",     "Webcam / livestream adult spam",                          0.80, r"\b(webcam\s+(model|show|live)|cam\s+(girl|show|site))\b"),
]

# Compile all rules once at import time
SPAM_RULES: list[SpamRule] = [
    SpamRule(
        name=name,
        category=cat,
        description=desc,
        weight=w,
        pattern=re.compile(pat, re.IGNORECASE | re.DOTALL),
    )
    for name, cat, desc, w, pat in _RAW
]

CATEGORIES: list[str] = sorted({r.category for r in SPAM_RULES})
RULE_COUNT: int = len(SPAM_RULES)


def match_rules(text: str) -> list[SpamRule]:
    """Run all rules against raw (uncleaned) text. Returns matched rules."""
    raw = str(text or "")
    return [r for r in SPAM_RULES if r.pattern.search(raw)]


def combined_keyword_boost(matched: list[SpamRule]) -> float:
    """Noisy-OR combination: boost = 1 - Π(1 - weight_i). Returns [0.0, 1.0]."""
    if not matched:
        return 0.0
    p = 1.0
    for r in matched:
        p *= (1.0 - r.weight)
    return min(1.0 - p, 0.9999)
