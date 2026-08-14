# API reference (summary)

Requests to the REST API must be authenticated with an API key sent in the
`Authorization: Bearer <key>` header. Keys are created and revoked from the
Developer Console under Settings → API Access.

Rate limits are enforced per API key: 60 requests per minute on the Free plan,
600 on Pro, and 6000 on Enterprise. When you exceed your bucket the API
returns HTTP 429 with a `Retry-After` header measured in seconds.

Uploaded files may be up to 100 MB each on Free, 2 GB on Pro, and 20 GB on
Enterprise. Files are virus-scanned before they become downloadable. Uploads
that fail scanning are quarantined and reported on the account activity log.

Webhooks can be registered for the events `job.completed`, `job.failed`,
`invoice.paid`, `invoice.failed`, and `alert.triggered`. Delivery is retried
with exponential backoff for up to 24 hours; the payload is signed with an
HMAC-SHA256 signature you can verify with your shared webhook secret.

Backups of all customer data are retained for 30 days on the Free plan and
90 days on Pro and Enterprise. Point-in-time restore is available on
Enterprise with an RPO of 5 minutes.
