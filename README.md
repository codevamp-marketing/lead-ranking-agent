# F2Fintech — Lead Ranking Agent

Production-grade AI marketing automation agent. Listens for new leads via Postgres LISTEN/NOTIFY, scores them using a two-layer engine, and writes AI fields back to the CRM — all in real-time, sub-second.

---

## Architecture

```
Lead Form Submission
        │
        ▼
  Supabase (Lead INSERT)
        │
        ▼  pg_notify("new_lead", json_payload)
  Postgres Trigger
        │
        ▼
  Agent: listen_loop()          ← always running, zero-poll
        │
        ├── score_lead()        ← Layer 1: ScoringRule table
        │                          Layer 2: Signal heuristics
        │
        ├── classify_lead()     ← Hot / Warm / Cold
        ├── next_best_action()  ← Call & Close / Follow-up / Nurture
        ├── predict_ltv()       ← Estimated fee revenue (INR)
        │
        ├── PATCH /update-lead/:id   ← CRM REST API
        │     └── Fallback: Supabase direct PATCH
        │
        ├── INSERT Activity (AI_Insight)   ← audit trail
        └── INSERT Notification            ← counsellor alert
```

---

## Scoring Logic

### Layer 1 — Admin-configurable rules (ScoringRule table)

| ruleType | ruleKey example | Effect |
|----------|----------------|--------|
| source | google ads | +baseScore × weight when source matches |
| campaign | diwali | +points when campaign field matches |
| tag | high intent | +points when any tag matches |

Admins can add/edit/disable rules in the CRM portal. No code change needed.

### Layer 2 — Signal heuristics (always active)

| Signal | Points |
|--------|--------|
| Source: Google_Ads | 25 |
| Source: Referral | 22 |
| Source: LinkedIn | 20 |
| Source: Website | 20 |
| Source: Facebook_Ads | 18 |
| Source: Instagram | 15 |
| Source: Manual | 10 |
| Course: Tier-1 (MBA/PGDM/MCA/M.Tech) | 20 |
| Course: Tier-2 (BBA/BCA/B.Tech) | 12 |
| Phone + Email both present | 15 |
| Phone only | 10 |
| Specialization filled | 8 |
| gclid/fbclid in attribution JSON | 5 |

### Classification thresholds

| Score | Type | Next Action |
|-------|------|-------------|
| ≥ 60 | 🔥 Hot | Call & Close — within 1 hour |
| 35–59 | 🌡 Warm | Follow-up Call — within 24 hours |
| < 35 | ❄ Cold | Email Nurturing — 7-day drip |

### Predicted LTV formula

```
LTV = ₹2,00,000 × course_multiplier × (score / 100)
      course_multiplier: Tier-1 = 1.5 | Tier-2 = 1.0 | Other = 0.7
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your values
```

**.env.example**
```env
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
CRM_API_BASE=http://localhost:3000/api
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-...supabase.com:5432/postgres
```

> **DATABASE_URL**: Get from Supabase Dashboard → Settings → Database → Connection String → URI → Session mode (port 5432)

### 3. Set up Postgres trigger (one-time)

Run `sql/01_notify_trigger.sql` in Supabase SQL Editor.

### 4. Run the agent

```bash
python -m agent.lead_ranking_agent
```

---

## Project structure

```
lead_agent/
├── agent/
│   └── lead_ranking_agent.py   # Main event loop, orchestration
├── scoring/
│   └── engine.py               # Two-layer scoring engine
├── config/
│   └── settings.py             # Pydantic-validated env config
├── utils/
│   └── logger.py               # Structured JSON logger
├── sql/
│   └── 01_notify_trigger.sql   # Postgres trigger (one-time setup)
├── requirements.txt
└── README.md
```

---

## Extending the agent

**Add a new scoring signal** → edit `scoring/engine.py` → `_signal_score()`

**Add a new rule type** → add a row in `ScoringRule` table (ruleType = your new type) + handle it in `_apply_db_rules()`

**Change LTV model** → edit `predict_ltv()` in `scoring/engine.py`

**Change Hot/Warm/Cold thresholds** → edit `classify_lead()` in `scoring/engine.py`

---

## Logs

All logs are JSON-structured for easy aggregation:

```json
{"time": "2025-01-15T10:22:01Z", "level": "INFO", "event": "lead_ranked",
 "lead_id": "abc-123", "score": 72, "type": "Hot", "action": "Call & Close — within 1 hour", "ltv": 216000.0}
```