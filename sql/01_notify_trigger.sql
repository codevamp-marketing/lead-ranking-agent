-- ═══════════════════════════════════════════════════════════════════════════
--  F2Fintech — Postgres LISTEN/NOTIFY trigger
--  Run in Supabase SQL Editor. Safe to re-run (CREATE OR REPLACE).
--  Aligned to CURRENT Lead schema: pickedBy + createdBy (ownerId removed).
-- ═══════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION notify_new_lead()
RETURNS trigger AS $$
DECLARE
  payload TEXT;
BEGIN
  -- Explicit field list avoids the 8KB NOTIFY limit (no row_to_json(NEW))
  -- camelCase columns must be quoted: "pickedBy", "createdBy"
  SELECT json_build_object(
    'id',             NEW.id,
    'name',           NEW.name,
    'email',          NEW.email,
    'phone',          NEW.phone,
    'course',         NEW.course,
    'specialization', NEW.specialization,
    'company',        NEW.company,
    'source',         NEW.source,
    'campaign',       NEW.campaign,
    'tags',           NEW.tags,
    'attribution',    NEW.attribution,
    'pickedBy',       NEW."pickedBy",
    'createdBy',      NEW."createdBy"
  )::TEXT INTO payload;

  PERFORM pg_notify('new_lead', payload);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Keep your existing trigger name
DROP TRIGGER IF EXISTS lead_inserted ON "Lead";
CREATE TRIGGER lead_inserted
AFTER INSERT ON "Lead"
FOR EACH ROW EXECUTE FUNCTION notify_new_lead();

-- ── Verify ───────────────────────────────────────────────────────────────────
-- SELECT trigger_name, event_manipulation, event_object_table
-- FROM information_schema.triggers
-- WHERE trigger_name = 'lead_inserted';