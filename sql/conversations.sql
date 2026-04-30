-- ============================================================
--  F2Fintech — Conversation Tables
--  Run ONCE in Supabase SQL Editor.
--  Required by webhook/conversation_store.py to persist messages.
-- ============================================================

-- Conversation: one row per lead (keyed by E.164 phone number)
CREATE TABLE IF NOT EXISTS "Conversation" (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    "leadId"    TEXT        NOT NULL UNIQUE,
    "leadName"  TEXT,
    "createdAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
    "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ConversationMessage: one row per message (inbound or outbound)
CREATE TABLE IF NOT EXISTS "ConversationMessage" (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    "conversationId"  UUID        NOT NULL REFERENCES "Conversation"(id) ON DELETE CASCADE,
    direction         TEXT        NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    body              TEXT        NOT NULL,
    channel           TEXT        NOT NULL DEFAULT 'whatsapp',
    "twilioSid"       TEXT,
    "createdAt"       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Fast history lookup index
CREATE INDEX IF NOT EXISTS idx_conv_message_conversation_id
    ON "ConversationMessage" ("conversationId", "createdAt");

-- Verify after running:
SELECT table_name FROM information_schema.tables
WHERE table_name IN ('Conversation', 'ConversationMessage');




#pip uninstall supabase -y
#pip install supabase==1.0.3