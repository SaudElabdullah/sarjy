# Vendored third-party assets

## supabase.js

- Package: `@supabase/supabase-js`
- Pinned version: `2.112.3` (resolved from the `@2` dist-tag at fetch time via
  `https://data.jsdelivr.com/v1/packages/npm/@supabase/supabase-js/resolved?specifier=2`)
- Source: `https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js`
- SHA-256: `ec004176d101aec77aeef266aa1c94411287fe2039c65ea5f6c72f5e14b3847d`
- Fetched: 2026-08-22

Vendored locally (rather than loaded from the jsdelivr CDN at request time) so the
page's CSP `script-src` can be `'self'` only — see `src/sarjy/interfaces/http/security.py`.

To update: bump the pinned version above, then re-fetch and re-verify:

```sh
curl -o src/sarjy/interfaces/web/static/supabase.js \
  https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.<new-version>/dist/umd/supabase.js
shasum -a 256 src/sarjy/interfaces/web/static/supabase.js
```

Record the new version and hash here.
