-- ============================================================
-- F2Fintech Lead Ranking Agent — Supabase SQL Setup
-- ============================================================
-- Run this ONCE in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- STEP 1: Create the function that fires the notify
-- It serialises the inserted Lead row as JSON and broadcasts
-- it on the 'new_lead' channel. The Python agent listens here.
CREATE OR REPLACE FUNCTION notify_new_lead()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('new_lead', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- STEP 2: Create the trigger on the "Lead" table
-- Fires AFTER every INSERT, calls the function above.
-- DROP IF EXISTS so re-running this script is safe.
DROP TRIGGER IF EXISTS lead_inserted ON "Lead";
CREATE TRIGGER lead_inserted
AFTER INSERT ON "Lead"
FOR EACH ROW EXECUTE FUNCTION notify_new_lead();

-- ============================================================
-- VERIFICATION (run after the trigger is created)
-- Watch your Python agent terminal — it should fire within 1s
-- ============================================================
-- INSERT INTO "Lead" (
--   id, name, email,
--   "createdAt", "lastInteraction",
--   score, "aiScore", "predictedLTV", "dealValue", tags
-- )
-- VALUES (
--   gen_random_uuid(), 'Test Lead', 'test@f2fintech.com',
--   now(), now(),
--   0, 0, 0, 0, '{}'
-- );
