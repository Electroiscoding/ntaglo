# Feed Ranking API

Send one user and a list of posts. Get the posts back ranked, with a score breakdown.

- `algo.py` is the original algorithm and is never modified.
- `bootstrap.py` reads the engine and API code out of `algo.py` at startup.
- `server.py` exposes it over HTTP with FastAPI.
- No Gradle, no Android build, no Colab. The server is plain Python in Docker, and any client (Android, iOS, web) can call it over HTTPS.

## Endpoints

| Method | Path        | Purpose                               | Auth                 |
|--------|-------------|---------------------------------------|----------------------|
| GET    | /health     | Liveness check                        | none                 |
| POST   | /v1/rank    | Rank posts for a user                 | X-API-Key if enabled |
| GET    | /v1/stats   | Request count, mean latency, cache    | X-API-Key if enabled |
| GET    | /docs       | Interactive Swagger UI                | none                 |
| GET    | /openapi.json | Machine-readable schema             | none                 |

## Input: POST /v1/rank

```json
{
  "user": { },
  "posts": [ ],
  "events": [ ],
  "trendingTexts": [ ],
  "tasteWeight": 0.3,
  "roadmapWeight": 0.25,
  "topK": 20,
  "enforceExposureCap": true
}
```

Only `user.userId` (optional) and `posts` (required, 1 to 2000) are needed. Everything else has a default.

### user
| Field | Type | Default | Meaning |
|---|---|---|---|
| userId | string | "anonymous" | Identifies the viewer (used for exposure capping) |
| accessibilityScore | 0 to 1 | 0.5 | Accessibility bias term |
| gpsLatitude / gpsLongitude | number | null | Viewer location; tiny boost for nearby posts |
| keyboardConcepts | string[] | [] | Concepts the user types; raises starting quality slightly |
| dwellHistorySeconds | number[] | [] | Past dwell times, used to rank the current dwell |
| deviceAgeYears | 0 to 30 | 2.0 | Old devices favor light posts |
| deviceIsCharging | bool | false | Charging devices favor heavy media |
| deviceBandwidthMbps | number | 10.0 | Low bandwidth favors light posts |
| roadmap | string or null | null | The user's goal in plain text, up to 2000 chars |
| tastes | string[] | [] | Interests, up to 50 |

### posts (1 to 2000 items)
| Field | Type | Default | Meaning |
|---|---|---|---|
| postId | string | required | Unique id, returned in the output |
| text | string | required | Post content (1 to 20000 chars) |
| authorId | string | "unknown" | Author id |
| hasArtifact | bool | false | Post has an attached artifact (bonus) |
| ageHours | number | 1.0 | Hours since posting |
| isVideo | bool | false | Video post |
| videoLengthSeconds | number | 0 | Video length |
| grammarPenalty | 0 to 1 | 0 | Quality penalty |
| metadataScore | 0 to 1 | 0.3 | Metadata quality |
| impressionCount | int | 0 | Views so far (exploration) |
| likeCount, replyCount | int | 0 | Counters |
| latitude / longitude | number | null | Post location |
| authorPostTexts | string[] | [] | Author's other posts (roadmap match), up to 50 |

### events (optional view history for this user)
| Field | Type | Default | Meaning |
|---|---|---|---|
| postId | string | required | Must match a post |
| dwellSeconds | number | required | Time spent on the post |
| viewportVisibleFraction | 0 to 1 | 1.0 | How much of the post was visible |
| watchedFraction | 0 to 1 | null | Video fraction watched |
| liked / replied | bool | false | Interactions |
| sessionSeconds | number | 60 | Session length |
| replyText | string | null | Reply text (negative language lowers score) |

### Other top-level fields
| Field | Default | Meaning |
|---|---|---|
| trendingTexts | built-in list | Texts that define "trending" |
| tasteWeight | 0.3 | 0 to 2, strength of taste matching |
| roadmapWeight | 0.25 | 0 to 2, strength of roadmap matching |
| topK | 20 | How many ranked posts to return |
| enforceExposureCap | true | Limits how often low-quality posts are shown to the same user |

## Output

```json
{
  "userId": "u1",
  "count": 2,
  "latencyMs": 41.7,
  "items": [
    {
      "rank": 1,
      "postId": "p1",
      "finalScore": 0.31,
      "engagementQuality": 0.72,
      "regencyScore": 0.05,
      "masteryScore": 1.1,
      "explorationScore": 0.3,
      "roadmapScore": 0.8,
      "tasteScore": 0.9,
      "exposureCap": 1.0
    }
  ]
}
```

- `items` is sorted best first. Sort by `rank` or `finalScore`.
- `exposureCap` of 0.05 means the post was judged low quality and is shown to a given user at most about 5 percent of the time.
- A `finalScore` of 0 can mean the exposure cap suppressed that post for this request.
- Errors: `422` for invalid input (the body explains which field), `401` for a wrong API key.

## Example

```bash
curl -X POST https://YOUR-URL/v1/rank \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "user": {"userId": "u1", "roadmap": "learn python", "tastes": ["python programming"]},
    "posts": [
      {"postId": "p1", "text": "python tutorial for beginners", "ageHours": 2},
      {"postId": "p2", "text": "grilled fish with lemon", "ageHours": 2}
    ],
    "events": [{"postId": "p1", "dwellSeconds": 30, "liked": true}],
    "topK": 5
  }'
```

## Deploy

Configuration (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| PORT | 8000 | Set automatically by most hosts |
| API_KEY | empty | If set, clients must send `X-API-Key`. Empty means fully public |
| CORS_ORIGINS | * | Comma-separated allowed web origins |
| LOG_LEVEL | INFO | Logging level |

### Option A: Render
1. Render dashboard, **New → Web Service**, connect this GitHub repo.
2. Runtime: **Docker**. Instance: at least 1 GB RAM (torch and the embedding model need it).
3. Health check path: `/health`.
4. Add `API_KEY` under Environment (or leave it out for a fully public API).
5. Deploy. Your URL is `https://<name>.onrender.com`, and docs are at `/docs`.

### Option B: Railway or Fly.io
Connect the repo, choose the Dockerfile, set the same variables. The `PORT` variable is handled automatically.

### Option C: Any VPS
```bash
git clone <your-repo-url> && cd <repo>
docker build -t ranking-api .
docker run -d --restart unless-stopped -p 80:8000 -e API_KEY=yourkey ranking-api
```

### Run locally
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn server:app --port 8000
```

## Operating notes

- Run a single worker (already set). The exposure-cap history and embedding cache live in memory and are shared safely by threads. If you later scale to several instances, move that state to Redis.
- The first start takes a few seconds while the model loads and warms up. `/health` reports `ready`.
- To update the algorithm, edit `algo.py` and commit. The next deploy picks it up.
- Put the service behind your host's HTTPS (Render, Railway and Fly provide it automatically).
