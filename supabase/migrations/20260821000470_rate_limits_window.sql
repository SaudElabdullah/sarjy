-- I2/I3: rate_limits held one row per (user_id, window_start), with the
-- 10-minute bucket and the day bucket writing into the SAME keyspace.
--
-- I2 (collision): at midnight the two bucket starts are the same timestamp, so
-- the day counter and the 10-minute counter collided into one row — each hit
-- incrementing it twice, and a user hitting the 10-minute limit at 00:00 also
-- burning through the daily allowance ten times faster.
--
-- I3 (boundary burst): a fixed 10-minute window resets to zero on the boundary,
-- so `limit` requests at 09:59 followed by `limit` more at 10:01 were all
-- allowed — twice the intended rate in two minutes. The fix is an approximate
-- sliding window: 5-minute buckets, with the check summing the current and the
-- previous one. That needs the two window kinds to be distinguishable, which is
-- what this column is for.
--
-- "window" is a reserved word in Postgres, so it is quoted everywhere.
alter table public.rate_limits add column "window" text not null default '10m';
alter table public.rate_limits drop constraint rate_limits_pkey;
alter table public.rate_limits add primary key (user_id, "window", window_start);
