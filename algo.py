import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "numpy", "pandas", "matplotlib", "pytest", "gradio", "pydantic"], check=True)
import os, sys
for name in ("contentRankingEngine", "rankingApi", "testContentRankingEngine"):
    sys.modules.pop(name, None)
    if os.path.exists(name + ".py"):
        os.remove(name + ".py")
engineSource = r'''
import math
import random
import re
import numpy as np
import sentence_transformers as sentenceTransformers
from dataclasses import dataclass
from typing import List, Optional

gpsWeight = 0.03
lowQualityGarbageThreshold = 0.001
lowQualityFloorThreshold = 0.10
exposureCapForLowQuality = 0.05
startingQualityFloor = 0.45
startingQualityCeiling = 0.95
repeatImpressionCapFraction = 0.45
defaultReachExponent = 1.5
averageSilentReadingWordsPerMinute = 238.0
textReadingTheta = 0.15
funKeyDensityCap = 0.45
negativityPenaltyCap = 0.45
minVisibleFractionDefault = 0.50
minVisibleSecondsDefault = 1.0
roadmapWeight = 0.25
roadmapMinimumSimilarity = 0.20
roadmapPostShare = 0.65
roadmapAuthorShare = 0.35
priorWeightMinimum = 5.0
priorWeightMaximum = 60.0
explorationNoiseMaximum = 0.05
explorationAlphaDefault = 0.10
maxViewsNormalizerDefault = 10000.0
locationDecayKm = 500.0
tasteWeightDefault = 0.30


@dataclass
class PostRecord:
    postId: str
    authorId: str
    text: str
    hasArtifact: bool
    createdAtEpoch: float
    isVideo: bool
    videoLengthSeconds: float
    grammarPenalty: float
    metadataScore: float
    impressionCount: int
    likeCount: int
    replyCount: int
    latitude: Optional[float]
    longitude: Optional[float]
    authorPostTexts: List[str]


@dataclass
class ViewEvent:
    postId: str
    userId: str
    dwellSeconds: float
    viewportVisibleFraction: float
    watchedFraction: Optional[float]
    liked: bool
    replied: bool
    sessionSeconds: float
    replyText: Optional[str]


@dataclass
class UserContext:
    userId: str
    accessibilityScore: float
    gpsLatitude: Optional[float]
    gpsLongitude: Optional[float]
    keyboardConcepts: List[str]
    dwellHistorySeconds: List[float]
    deviceAgeYears: float
    deviceIsCharging: bool
    deviceBandwidthMbps: float
    roadmap: Optional[str]
    tastes: List[str]


@dataclass
class RankedPost:
    postId: str
    finalScore: float
    engagementQuality: float
    regencyScore: float
    masteryScore: float
    explorationScore: float
    roadmapScore: float
    tasteScore: float
    exposureCap: float


class EmbeddingEngine:
    def __init__(self, modelName="all-MiniLM-L6-v2"):
        self.model = sentenceTransformers.SentenceTransformer(modelName)
        self.cache = {}

    def embed(self, text):
        if text in self.cache:
            return self.cache[text]
        raw = self.model.encode(text)
        vector = np.asarray(raw, dtype=np.float32)
        normalized = vector / (np.linalg.norm(vector) + 1e-9)
        self.cache[text] = normalized
        return normalized

    def embedMany(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            raw = self.model.encode(missing)
            for text, row in zip(missing, raw):
                vector = np.asarray(row, dtype=np.float32)
                self.cache[text] = vector / (np.linalg.norm(vector) + 1e-9)
        return [self.cache[text] for text in texts]


def clip(value, low, high):
    return max(low, min(high, value))


def cosineSimilarity(vectorA, vectorB):
    denom = (np.linalg.norm(vectorA) * np.linalg.norm(vectorB)) + 1e-9
    return float(np.dot(vectorA, vectorB) / denom)


def computeTrendCentroid(embeddingEngine, trendingTexts):
    vectors = embeddingEngine.embedMany(trendingTexts)
    return np.mean(np.vstack(vectors), axis=0)


def piecewiseInterpolate(x, anchors):
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for index in range(len(anchors) - 1):
        xLow, yLow = anchors[index]
        xHigh, yHigh = anchors[index + 1]
        if xLow <= x <= xHigh:
            t = (x - xLow) / (xHigh - xLow)
            return yLow + t * (yHigh - yLow)
    return anchors[-1][1]


def haversineKm(latA, lonA, latB, lonB):
    radiusKm = 6371.0
    phiA = math.radians(latA)
    phiB = math.radians(latB)
    deltaPhi = math.radians(latB - latA)
    deltaLambda = math.radians(lonB - lonA)
    a = math.sin(deltaPhi / 2) ** 2 + math.cos(phiA) * math.cos(phiB) * math.sin(deltaLambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radiusKm * c


def locationAffinityScore(userContext, post):
    if userContext.gpsLatitude is None or userContext.gpsLongitude is None:
        return 0.0
    if post.latitude is None or post.longitude is None:
        return 0.0
    distanceKm = haversineKm(userContext.gpsLatitude, userContext.gpsLongitude, post.latitude, post.longitude)
    return gpsWeight * math.exp(-distanceKm / locationDecayKm)


def durabilityDensity(dwellSeconds, historyDwellSeconds):
    sample = sorted(historyDwellSeconds + [dwellSeconds])
    total = len(sample)
    if total <= 1:
        return 1.0
    rank = sum(1 for value in sample if value <= dwellSeconds)
    return rank / total


def accessRegencyScore(ageHours, reachExponent, repeatImpressionsRecent):
    base = (1.0 / (4.0 + ageHours)) ** reachExponent
    if repeatImpressionsRecent >= 3:
        dampFactor = repeatImpressionCapFraction / max(repeatImpressionsRecent / 3.0, 1.0)
        base = base * dampFactor
    return base

def softRegency(ageHours, repeatImpressionsRecent):
    base = 1.0 / (1.0 + (ageHours / 48.0) ** 1.2)
    if repeatImpressionsRecent >= 3:
        base = base * (repeatImpressionCapFraction / max(repeatImpressionsRecent / 3.0, 1.0))
    return base

def squash(value, midpoint, spread):
    return 1.0 / (1.0 + math.exp(-(value - midpoint) / spread))

def calibrateSimilarity(similarity, floor=0.15, ceiling=0.55):
    return clip((similarity - floor) / (ceiling - floor), 0.0, 1.0)

def sessionActiveTimeScore(viewEvent, post):
    if post.isVideo and post.videoLengthSeconds > 0:
        if viewEvent.watchedFraction is not None:
            fraction = viewEvent.watchedFraction
        else:
            fraction = viewEvent.dwellSeconds / post.videoLengthSeconds
        fraction = clip(fraction, 0.0, 1.0)
        anchors = [(0.05, 0.01), (0.25, 0.20), (0.50, 0.50), (0.75, 0.70), (1.00, 0.10)]
        return piecewiseInterpolate(fraction, anchors)
    wordCount = len(post.text.split())
    if wordCount > 50:
        expectedReadSeconds = (wordCount / averageSilentReadingWordsPerMinute) * 60.0
        ratio = clip(viewEvent.dwellSeconds / max(expectedReadSeconds, 1.0), 0.0, 2.0)
        return textReadingTheta * ratio
    return clip(viewEvent.dwellSeconds / 15.0, 0.0, 1.0)


def contentQualityScore(post, embeddingEngine, trendCentroid):
    metadataComponent = clip(post.metadataScore, 0.0, 0.5) * (1.25 / 0.5)
    vector = embeddingEngine.embed(post.text)
    relevance = cosineSimilarity(vector, trendCentroid)
    clusterComponent = clip((relevance + 1.0) / 2.0, 0.0, 1.0) * 0.75
    artifactBonus = 0.20 if post.hasArtifact else 0.0
    raw = 0.55 * metadataComponent + 0.45 * clusterComponent + artifactBonus - post.grammarPenalty
    return clip(raw, 0.0, 1.5)


def bayesianStartingQuality(rawQuality, artifactBonus, funKeyDensity, grammarPenalty):
    combined = rawQuality + artifactBonus + clip(funKeyDensity, 0.0, funKeyDensityCap) - grammarPenalty
    return clip(combined, startingQualityFloor, startingQualityCeiling)


def dynamicPriorWeight(post):
    if post.isVideo and post.videoLengthSeconds > 0:
        return clip(post.videoLengthSeconds / 2.0, priorWeightMinimum, priorWeightMaximum)
    wordCount = len(post.text.split())
    expectedReadSeconds = (wordCount / averageSilentReadingWordsPerMinute) * 60.0
    return clip(expectedReadSeconds / 2.0, priorWeightMinimum, priorWeightMaximum)


def collectPositiveSignals(viewEvents, minVisibleFraction=minVisibleFractionDefault, minVisibleSeconds=minVisibleSecondsDefault):
    seenPosts = set()
    signalTotal = 0.0
    impressions = 0
    for event in viewEvents:
        eligible = event.viewportVisibleFraction >= minVisibleFraction and event.dwellSeconds >= minVisibleSeconds
        if not eligible:
            continue
        dedupKey = (event.userId, event.postId)
        if dedupKey in seenPosts:
            continue
        seenPosts.add(dedupKey)
        impressions += 1
        if event.dwellSeconds > 15.0:
            signalTotal += 1.0
        elif event.dwellSeconds >= 5.0:
            signalTotal += 0.2
        elif event.dwellSeconds < 1.5:
            signalTotal -= 0.10
        if event.liked:
            signalTotal += 0.15
        if event.replied:
            signalTotal += 1.0
    return signalTotal, impressions


def combinedEngagementQuality(startingQuality, signalTotal, impressions, priorWeight):
    denominator = priorWeight + impressions
    if denominator <= 0:
        return startingQuality
    return (priorWeight * startingQuality + signalTotal) / denominator


def exposureCapFraction(qualityScore):
    if qualityScore <= lowQualityGarbageThreshold or qualityScore < lowQualityFloorThreshold:
        return exposureCapForLowQuality
    return 1.0


def engineeringScore(biasTerm, retentionTerm):
    return 0.5 * biasTerm + 1.2 * retentionTerm


def timeMachineScore(dwellSeconds, sessionSeconds):
    if sessionSeconds <= 0:
        return 0.0
    return clip(dwellSeconds / sessionSeconds, 0.0, 1.0)


def abiLearningScore(topicDistanceDays):
    base = 1.0 + 0.3 / (1.0 + (topicDistanceDays / 30.0) ** 2)
    if topicDistanceDays < 10:
        base = base * 1.20
    return base


rejectionPatterns = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [r"\bno\b", r"\bnever\b", r"\bhate\b", r"\bboring\b", r"\bworst\b", r"\bdislike\b"]
]


def rejectionLanguageScore(replyText):
    if not replyText:
        return 1.0
    hits = sum(1 for pattern in rejectionPatterns if pattern.search(replyText))
    penalty = clip(hits * 0.10, 0.0, negativityPenaltyCap)
    return 1.0 - penalty


def deviceTrendScore(deviceAgeYears, deviceIsCharging, deviceBandwidthMbps, postIsHeavyMedia):
    score = 1.0
    if deviceAgeYears >= 8.0 or deviceBandwidthMbps < 0.7:
        score = score * 0.80 if postIsHeavyMedia else score * 1.10
    if deviceIsCharging and deviceBandwidthMbps >= 0.8 and postIsHeavyMedia:
        score = score * 1.10
    return score


def finalMasteryScore(scoreE, scoreM, scoreL, scoreX, scoreG):
    return scoreE * scoreM * scoreL * scoreX * scoreG


def explorationScore(post, trendEmbedding, embeddingEngine, viewsSoFar, maxViewsNormalizer=maxViewsNormalizerDefault, explorationAlpha=explorationAlphaDefault, randomSeed=None, qualityHint=1.0):
    rng = random.Random(randomSeed)
    viewRatio = clip(viewsSoFar / max(maxViewsNormalizer, 1.0), 0.0, 1.0)
    trendBoost = 1.0 + explorationAlpha * viewRatio
    postVector = embeddingEngine.embed(post.text)
    trendSimilarity = clip((cosineSimilarity(postVector, trendEmbedding) + 1.0) / 2.0, 0.0, 1.0)
    uniformNoise = rng.uniform(0.0, explorationNoiseMaximum)
    return (trendBoost * trendSimilarity * 0.5 + uniformNoise) * clip(qualityHint, 0.0, 1.0)


def authorEmbedding(embeddingEngine, authorPostTexts):
    if not authorPostTexts:
        return None
    vectors = embeddingEngine.embedMany(authorPostTexts)
    mean = np.mean(np.vstack(vectors), axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def roadmapAlignmentScore(embeddingEngine, roadmap, post):
    if roadmap is None or not roadmap.strip():
        return 0.0
    roadmapVector = embeddingEngine.embed(roadmap)
    postSimilarity = cosineSimilarity(roadmapVector, embeddingEngine.embed(post.text))
    authorVector = authorEmbedding(embeddingEngine, post.authorPostTexts)
    if authorVector is None:
        combined = postSimilarity
    else:
        authorSimilarity = cosineSimilarity(roadmapVector, authorVector)
        combined = roadmapPostShare * postSimilarity + roadmapAuthorShare * authorSimilarity
    return calibrateSimilarity(combined)

def tasteAlignmentScore(embeddingEngine, tastes, post):
    cleaned = [taste.strip() for taste in tastes if taste and taste.strip()]
    if not cleaned:
        return 0.0
    postVector = embeddingEngine.embed(post.text)
    similarities = [cosineSimilarity(vector, postVector) for vector in embeddingEngine.embedMany(cleaned)]
    best = max(similarities)
    return calibrateSimilarity(best)

class ExposureGovernor:
    def __init__(self):
        self.history = {}

    def allow(self, userId, postId, capFraction):
        key = (userId, postId)
        shown, total = self.history.get(key, (0, 0))
        total += 1
        projectedShown = shown + 1
        projectedRate = projectedShown / total
        if capFraction >= 1.0 or projectedRate <= capFraction:
            shown = projectedShown
            allowed = True
        else:
            allowed = False
        self.history[key] = (shown, total)
        return allowed


def rankFeedForUser(userContext, posts, viewEventsByPost, embeddingEngine, trendCentroid, trendEmbedding, currentEpoch, governor=None, tasteWeight=tasteWeightDefault, roadmapWeightValue=roadmapWeight):
    texts = [post.text for post in posts]
    for post in posts:
        texts.extend(post.authorPostTexts)
    if userContext.roadmap:
        texts.append(userContext.roadmap)
    texts.extend([taste for taste in userContext.tastes if taste and taste.strip()])
    embeddingEngine.embedMany(texts)
    results = []
    for post in posts:
        events = viewEventsByPost.get(post.postId, [])
        latestEvent = events[-1] if events else None
        rawQuality = contentQualityScore(post, embeddingEngine, trendCentroid)
        ageHours = max((currentEpoch - post.createdAtEpoch) / 3600.0, 0.0)
        repeatImpressions = len(events)
        regency = accessRegencyScore(ageHours, defaultReachExponent, repeatImpressions)
        density = durabilityDensity(latestEvent.dwellSeconds, userContext.dwellHistorySeconds) if latestEvent else 0.0
        location = locationAffinityScore(userContext, post)
        exploration = explorationScore(post, trendEmbedding, embeddingEngine, post.impressionCount, qualityHint=clip(post.metadataScore * 2.0 - post.grammarPenalty, 0.0, 1.0))
        roadmap = roadmapAlignmentScore(embeddingEngine, userContext.roadmap, post)
        taste = tasteAlignmentScore(embeddingEngine, userContext.tastes, post)
        artifactBonus = 0.20 if post.hasArtifact else 0.0
        funKeyDensity = clip(len(userContext.keyboardConcepts) / 100.0, 0.0, funKeyDensityCap)
        startingQuality = bayesianStartingQuality(rawQuality, artifactBonus, funKeyDensity, post.grammarPenalty)
        signalTotal, impressions = collectPositiveSignals(events)
        priorWeight = dynamicPriorWeight(post)
        engagementQuality = combinedEngagementQuality(startingQuality, signalTotal, impressions, priorWeight)
        exposureCap = exposureCapFraction(engagementQuality)
        biasTerm = clip(userContext.accessibilityScore, 0.0, 1.0)
        scoreE = engineeringScore(biasTerm, density)
        dwellForTime = latestEvent.dwellSeconds if latestEvent else 0.0
        sessionForTime = latestEvent.sessionSeconds if latestEvent else 1.0
        scoreM = timeMachineScore(dwellForTime, sessionForTime)
        scoreL = abiLearningScore(ageHours / 24.0)
        replyText = latestEvent.replyText if latestEvent else None
        scoreX = rejectionLanguageScore(replyText)
        scoreG = deviceTrendScore(userContext.deviceAgeYears, userContext.deviceIsCharging, userContext.deviceBandwidthMbps, post.isVideo)
        if latestEvent is None:
            scoreE = engineeringScore(biasTerm, 0.5)
            scoreM = 0.5
        mastery = finalMasteryScore(scoreE, scoreM, scoreL, scoreX, scoreG)
        freshness = softRegency(ageHours, repeatImpressions)
        masteryNorm = squash(mastery, 1.0, 0.8)
        qualityNorm = clip(engagementQuality, 0.0, 1.0)
        exploreNorm = clip(exploration, 0.0, 1.0)
        additiveScore = 0.22 * qualityNorm + 0.15 * roadmapWeightValue * roadmap + 0.15 * tasteWeight * taste + 0.10 * masteryNorm + 0.08 * freshness + 0.03 * exploreNorm
        relevanceLift = 1.0 + 1.2 * clip(max(roadmap, taste), 0.0, 1.0)
        combinedRank = additiveScore * relevanceLift * (1.0 + location) * exposureCap
        combinedRank = additiveScore * (1.0 + location) * exposureCap
        if governor is not None:
            if not governor.allow(userContext.userId, post.postId, exposureCap):
                combinedRank = 0.0
        results.append(RankedPost(post.postId, combinedRank, engagementQuality, regency, mastery, exploration, roadmap, taste, exposureCap))
    results.sort(key=lambda entry: entry.finalScore, reverse=True)
    return results
'''

apiSource = r'''
import time
import threading
from typing import List, Optional
from pydantic import BaseModel, Field
import contentRankingEngine as engine


class PostInput(BaseModel):
    postId: str
    text: str = Field(min_length=1, max_length=20000)
    authorId: str = "unknown"
    hasArtifact: bool = False
    ageHours: float = Field(default=1.0, ge=0.0, le=100000.0)
    isVideo: bool = False
    videoLengthSeconds: float = Field(default=0.0, ge=0.0, le=86400.0)
    grammarPenalty: float = Field(default=0.0, ge=0.0, le=1.0)
    metadataScore: float = Field(default=0.3, ge=0.0, le=1.0)
    impressionCount: int = Field(default=0, ge=0)
    likeCount: int = Field(default=0, ge=0)
    replyCount: int = Field(default=0, ge=0)
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    authorPostTexts: List[str] = Field(default_factory=list, max_length=50)


class EventInput(BaseModel):
    postId: str
    dwellSeconds: float = Field(ge=0.0, le=86400.0)
    viewportVisibleFraction: float = Field(default=1.0, ge=0.0, le=1.0)
    watchedFraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    liked: bool = False
    replied: bool = False
    sessionSeconds: float = Field(default=60.0, ge=0.0, le=86400.0)
    replyText: Optional[str] = None


class UserInput(BaseModel):
    userId: str = "anonymous"
    accessibilityScore: float = Field(default=0.5, ge=0.0, le=1.0)
    gpsLatitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    gpsLongitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    keyboardConcepts: List[str] = Field(default_factory=list, max_length=200)
    dwellHistorySeconds: List[float] = Field(default_factory=list, max_length=1000)
    deviceAgeYears: float = Field(default=2.0, ge=0.0, le=30.0)
    deviceIsCharging: bool = False
    deviceBandwidthMbps: float = Field(default=10.0, ge=0.0, le=10000.0)
    roadmap: Optional[str] = Field(default=None, max_length=2000)
    tastes: List[str] = Field(default_factory=list, max_length=50)


class RankRequest(BaseModel):
    user: UserInput
    posts: List[PostInput] = Field(min_length=1, max_length=2000)
    events: List[EventInput] = Field(default_factory=list, max_length=20000)
    trendingTexts: List[str] = Field(default_factory=list, max_length=100)
    tasteWeight: float = Field(default=engine.tasteWeightDefault, ge=0.0, le=2.0)
    roadmapWeight: float = Field(default=engine.roadmapWeight, ge=0.0, le=2.0)
    topK: int = Field(default=20, ge=1, le=2000)
    enforceExposureCap: bool = True


class RankedItem(BaseModel):
    rank: int
    postId: str
    finalScore: float
    engagementQuality: float
    regencyScore: float
    masteryScore: float
    explorationScore: float
    roadmapScore: float
    tasteScore: float
    exposureCap: float


class RankResponse(BaseModel):
    userId: str
    count: int
    latencyMs: float
    items: List[RankedItem]


defaultTrendingTexts = [
    "python tutorial for beginners",
    "machine learning explained simply",
    "latest smartphone review",
    "easy home cooking recipes",
    "beginner fitness routine",
    "budget travel tips",
]


class RankingService:
    def __init__(self, embeddingEngine=None):
        self.embeddingEngine = embeddingEngine if embeddingEngine is not None else engine.EmbeddingEngine()
        self.governor = engine.ExposureGovernor()
        self.lock = threading.Lock()
        self.requestCount = 0
        self.totalLatencyMs = 0.0

    def rank(self, request):
        started = time.perf_counter()
        now = time.time()
        posts = [
            engine.PostRecord(
                item.postId, item.authorId, item.text, item.hasArtifact, now - item.ageHours * 3600.0,
                item.isVideo, item.videoLengthSeconds, item.grammarPenalty, item.metadataScore,
                item.impressionCount, item.likeCount, item.replyCount, item.latitude, item.longitude,
                list(item.authorPostTexts),
            )
            for item in request.posts
        ]
        eventsByPost = {}
        for item in request.events:
            eventsByPost.setdefault(item.postId, []).append(
                engine.ViewEvent(item.postId, request.user.userId, item.dwellSeconds, item.viewportVisibleFraction, item.watchedFraction, item.liked, item.replied, item.sessionSeconds, item.replyText)
            )
        userContext = engine.UserContext(
            request.user.userId, request.user.accessibilityScore, request.user.gpsLatitude, request.user.gpsLongitude,
            list(request.user.keyboardConcepts), list(request.user.dwellHistorySeconds), request.user.deviceAgeYears,
            request.user.deviceIsCharging, request.user.deviceBandwidthMbps, request.user.roadmap, list(request.user.tastes),
        )
        trendingTexts = request.trendingTexts if request.trendingTexts else defaultTrendingTexts
        trendCentroid = engine.computeTrendCentroid(self.embeddingEngine, trendingTexts)
        trendEmbedding = self.embeddingEngine.embed("trending topics today")
        governor = self.governor if request.enforceExposureCap else None
        with self.lock:
            ranked = engine.rankFeedForUser(userContext, posts, eventsByPost, self.embeddingEngine, trendCentroid, trendEmbedding, now, governor, request.tasteWeight, request.roadmapWeight)
        items = [
            RankedItem(rank=index + 1, postId=entry.postId, finalScore=entry.finalScore, engagementQuality=entry.engagementQuality, regencyScore=entry.regencyScore, masteryScore=entry.masteryScore, explorationScore=entry.explorationScore, roadmapScore=entry.roadmapScore, tasteScore=entry.tasteScore, exposureCap=entry.exposureCap)
            for index, entry in enumerate(ranked[: request.topK])
        ]
        latencyMs = (time.perf_counter() - started) * 1000.0
        self.requestCount += 1
        self.totalLatencyMs += latencyMs
        return RankResponse(userId=request.user.userId, count=len(items), latencyMs=latencyMs, items=items)

    def stats(self):
        meanLatency = self.totalLatencyMs / self.requestCount if self.requestCount else 0.0
        return {"requests": self.requestCount, "meanLatencyMs": meanLatency, "cachedTexts": len(self.embeddingEngine.cache)}


def rankJson(service, payload):
    request = RankRequest.model_validate(payload)
    return service.rank(request).model_dump()
'''

testSource = r'''
import time
import contentRankingEngine as engine
import rankingApi as api

embeddingEngine = engine.EmbeddingEngine()
service = api.RankingService(embeddingEngine)


def buildPost(postId, authorId, text, hasArtifact=False, ageSeconds=3600, isVideo=False, videoLengthSeconds=0.0, grammarPenalty=0.0, metadataScore=0.3, impressionCount=10, likeCount=0, replyCount=0, latitude=None, longitude=None, authorPostTexts=None):
    return engine.PostRecord(postId, authorId, text, hasArtifact, time.time() - ageSeconds, isVideo, videoLengthSeconds, grammarPenalty, metadataScore, impressionCount, likeCount, replyCount, latitude, longitude, authorPostTexts if authorPostTexts is not None else [])


def buildUser(userId="u1", accessibilityScore=0.5, gpsLatitude=None, gpsLongitude=None, keyboardConcepts=None, dwellHistorySeconds=None, deviceAgeYears=2.0, deviceIsCharging=False, deviceBandwidthMbps=5.0, roadmap=None, tastes=None):
    return engine.UserContext(userId, accessibilityScore, gpsLatitude, gpsLongitude, keyboardConcepts if keyboardConcepts is not None else [], dwellHistorySeconds if dwellHistorySeconds is not None else [], deviceAgeYears, deviceIsCharging, deviceBandwidthMbps, roadmap, tastes if tastes is not None else [])


def buildEvent(postId, userId="u1", dwellSeconds=10.0, visibleFraction=0.9, watchedFraction=None, liked=False, replied=False, sessionSeconds=60.0, replyText=None):
    return engine.ViewEvent(postId, userId, dwellSeconds, visibleFraction, watchedFraction, liked, replied, sessionSeconds, replyText)


def testClipBounds():
    assert engine.clip(5, 0, 1) == 1
    assert engine.clip(-5, 0, 1) == 0
    assert engine.clip(0.5, 0, 1) == 0.5


def testEmbeddingNormalizedAndCached():
    first = embeddingEngine.embed("caching check sentence")
    second = embeddingEngine.embed("caching check sentence")
    assert abs(float(sum(first * first)) - 1.0) < 1e-4
    assert first is second


def testEmbedManyMatchesEmbed():
    many = embeddingEngine.embedMany(["alpha text", "beta text", "alpha text"])
    assert many[0] is many[2]
    assert abs(engine.cosineSimilarity(many[0], embeddingEngine.embed("alpha text")) - 1.0) < 1e-4


def testEmbeddingSemanticOrdering():
    query = embeddingEngine.embed("python programming tutorial")
    near = embeddingEngine.embed("learn to code in python")
    far = embeddingEngine.embed("grilled fish with lemon and herbs")
    assert engine.cosineSimilarity(query, near) > engine.cosineSimilarity(query, far)


def testPiecewiseInterpolate():
    anchors = [(0.05, 0.01), (0.25, 0.20), (0.50, 0.50), (0.75, 0.70), (1.00, 0.10)]
    for x, y in anchors:
        assert abs(engine.piecewiseInterpolate(x, anchors) - y) < 1e-6
    assert abs(engine.piecewiseInterpolate(0.375, anchors) - 0.35) < 1e-6
    assert engine.piecewiseInterpolate(0.0, anchors) == 0.01
    assert engine.piecewiseInterpolate(2.0, anchors) == 0.10


def testHaversine():
    assert abs(engine.haversineKm(40.7128, -74.0060, 34.0522, -118.2437) - 3935) < 60
    assert engine.haversineKm(10.0, 20.0, 10.0, 20.0) == 0.0


def testLocationAffinity():
    user = buildUser(gpsLatitude=40.7128, gpsLongitude=-74.0060)
    near = buildPost("p1", "a1", "hello", latitude=40.7300, longitude=-74.0000)
    far = buildPost("p2", "a1", "hello", latitude=34.0522, longitude=-118.2437)
    assert engine.locationAffinityScore(user, near) <= engine.gpsWeight
    assert engine.locationAffinityScore(user, near) > engine.locationAffinityScore(user, far)
    assert engine.locationAffinityScore(buildUser(), near) == 0.0


def testDurabilityDensity():
    history = [5.0, 10.0, 15.0, 20.0, 25.0]
    assert abs(engine.durabilityDensity(30.0, history) - 1.0) < 1e-9
    assert abs(engine.durabilityDensity(1.0, history) - (1.0 / 6.0)) < 1e-9
    assert engine.durabilityDensity(10.0, []) == 1.0


def testAccessRegency():
    assert engine.accessRegencyScore(1.0, 1.5, 0) > engine.accessRegencyScore(100.0, 1.5, 0)
    assert engine.accessRegencyScore(10.0, 1.5, 10) < engine.accessRegencyScore(10.0, 1.5, 1)
    assert abs(engine.accessRegencyScore(6.0, 2.0, 0) - (1.0 / 10.0) ** 2.0) < 1e-12


def testSessionActiveTime():
    video = buildPost("v1", "a1", "video", isVideo=True, videoLengthSeconds=100.0)
    assert abs(engine.sessionActiveTimeScore(buildEvent("v1", watchedFraction=0.5), video) - 0.50) < 1e-9
    assert abs(engine.sessionActiveTimeScore(buildEvent("v1", dwellSeconds=25.0), video) - 0.20) < 1e-9
    assert engine.sessionActiveTimeScore(buildEvent("v1", watchedFraction=1.0), video) < engine.sessionActiveTimeScore(buildEvent("v1", watchedFraction=0.75), video)
    longPost = buildPost("t1", "a1", " ".join(["word"] * 238))
    assert abs(engine.sessionActiveTimeScore(buildEvent("t1", dwellSeconds=60.0), longPost) - engine.textReadingTheta) < 1e-9
    shortPost = buildPost("t2", "a1", "short text")
    assert abs(engine.sessionActiveTimeScore(buildEvent("t2", dwellSeconds=7.5), shortPost) - 0.5) < 1e-9


def testContentQuality():
    trend = engine.computeTrendCentroid(embeddingEngine, ["python tutorial for beginners", "how neural networks learn"])
    relevant = buildPost("p1", "a1", "python tutorial for beginners with neural networks")
    irrelevant = buildPost("p2", "a1", "sunset over the quiet harbour with sailboats")
    relevantScore = engine.contentQualityScore(relevant, embeddingEngine, trend)
    assert 0.0 <= relevantScore <= 1.5
    assert relevantScore > engine.contentQualityScore(irrelevant, embeddingEngine, trend)


def testBayesianStartingQuality():
    assert abs(engine.bayesianStartingQuality(0.0, 0.0, 0.0, 0.5) - engine.startingQualityFloor) < 1e-9
    assert abs(engine.bayesianStartingQuality(1.0, 0.20, 0.45, 0.0) - engine.startingQualityCeiling) < 1e-9
    assert abs(engine.bayesianStartingQuality(0.5, 0.0, 0.0, 0.0) - 0.5) < 1e-9


def testDynamicPriorWeight():
    assert abs(engine.dynamicPriorWeight(buildPost("v", "a", "video", isVideo=True, videoLengthSeconds=40.0)) - 20.0) < 1e-9
    assert engine.dynamicPriorWeight(buildPost("v", "a", "video", isVideo=True, videoLengthSeconds=2.0)) == engine.priorWeightMinimum
    assert engine.dynamicPriorWeight(buildPost("v", "a", "video", isVideo=True, videoLengthSeconds=3600.0)) == engine.priorWeightMaximum
    assert engine.dynamicPriorWeight(buildPost("t", "a", " ".join(["word"] * 476))) == engine.priorWeightMaximum


def testCollectPositiveSignals():
    events = [
        buildEvent("p1", dwellSeconds=20.0, visibleFraction=0.6),
        buildEvent("p1", dwellSeconds=20.0, visibleFraction=0.6),
        buildEvent("p2", dwellSeconds=6.0, visibleFraction=0.55),
        buildEvent("p3", dwellSeconds=1.0, visibleFraction=0.9),
        buildEvent("p4", dwellSeconds=3.0, visibleFraction=0.9, liked=True),
        buildEvent("p5", dwellSeconds=0.5, visibleFraction=0.9),
        buildEvent("p6", dwellSeconds=20.0, visibleFraction=0.3),
        buildEvent("p7", dwellSeconds=25.0, visibleFraction=0.8, replied=True),
    ]
    signalTotal, impressions = engine.collectPositiveSignals(events)
    assert impressions == 5
    assert abs(signalTotal - 3.25) < 1e-6
    assert engine.collectPositiveSignals([]) == (0.0, 0)


def testCombinedEngagementQuality():
    assert abs(engine.combinedEngagementQuality(0.6, 3.25, 5, 20.0) - 0.61) < 1e-6
    assert engine.combinedEngagementQuality(0.7, 0.0, 0, 0.0) == 0.7


def testExposureCap():
    assert engine.exposureCapFraction(0.0005) == engine.exposureCapForLowQuality
    assert engine.exposureCapFraction(0.05) == engine.exposureCapForLowQuality
    assert engine.exposureCapFraction(0.10) == 1.0
    assert engine.exposureCapFraction(0.5) == 1.0
    quality = engine.combinedEngagementQuality(0.45, -5.0, 50, 20.0)
    assert quality < engine.lowQualityFloorThreshold
    assert engine.exposureCapFraction(quality) == engine.exposureCapForLowQuality


def testMasteryComponents():
    assert abs(engine.engineeringScore(0.8, 0.5) - 1.0) < 1e-9
    assert abs(engine.timeMachineScore(30, 60) - 0.5) < 1e-9
    assert engine.timeMachineScore(90, 60) == 1.0
    assert engine.timeMachineScore(10, 0) == 0.0
    assert abs(engine.abiLearningScore(5) - 1.5503) < 0.01
    assert abs(engine.abiLearningScore(50) - 1.0794) < 0.01
    assert engine.finalMasteryScore(1, 2, 3, 4, 5) == 120


def testRejectionLanguage():
    assert abs(engine.rejectionLanguageScore("I hate this, never again") - 0.8) < 1e-9
    assert engine.rejectionLanguageScore(None) == 1.0
    assert engine.rejectionLanguageScore("great work, love it") == 1.0
    assert abs(engine.rejectionLanguageScore("no never hate boring worst dislike") - (1.0 - engine.negativityPenaltyCap)) < 1e-9


def testDeviceTrend():
    assert abs(engine.deviceTrendScore(9, False, 0.5, True) - 0.80) < 1e-9
    assert abs(engine.deviceTrendScore(2, True, 5.0, True) - 1.10) < 1e-9
    assert abs(engine.deviceTrendScore(9, False, 0.5, False) - 1.10) < 1e-9


def testExploration():
    trend = embeddingEngine.embed("trending topics today")
    post = buildPost("p1", "a1", "a post about trending topics")
    first = engine.explorationScore(post, trend, embeddingEngine, 100, randomSeed=7)
    second = engine.explorationScore(post, trend, embeddingEngine, 100, randomSeed=7)
    assert first == second
    assert 0.0 <= first <= 0.5 * (1.0 + engine.explorationAlphaDefault) + engine.explorationNoiseMaximum


def testAuthorEmbedding():
    assert engine.authorEmbedding(embeddingEngine, []) is None
    vector = engine.authorEmbedding(embeddingEngine, ["first post", "second post"])
    assert abs(float(sum(vector * vector)) - 1.0) < 1e-4


def testRoadmapAlignment():
    roadmap = "learn python programming and machine learning"
    aligned = buildPost("a", "x", "python machine learning tutorial for beginners", authorPostTexts=["intro to python", "neural network basics"])
    unaligned = buildPost("b", "y", "best street food recipes in the city", authorPostTexts=["restaurant reviews", "cooking pasta"])
    alignedScore = engine.roadmapAlignmentScore(embeddingEngine, roadmap, aligned)
    assert alignedScore > engine.roadmapAlignmentScore(embeddingEngine, roadmap, unaligned)
    assert 0.0 <= alignedScore <= 1.0
    assert engine.roadmapAlignmentScore(embeddingEngine, None, aligned) == 0.0
    assert engine.roadmapAlignmentScore(embeddingEngine, "   ", aligned) == 0.0


def testTasteAlignment():
    cooking = buildPost("c", "x", "easy weeknight pasta recipe with garlic")
    coding = buildPost("d", "x", "python decorators explained with code examples")
    assert engine.tasteAlignmentScore(embeddingEngine, ["cooking recipes"], cooking) > engine.tasteAlignmentScore(embeddingEngine, ["cooking recipes"], coding)
    assert engine.tasteAlignmentScore(embeddingEngine, [], cooking) == 0.0
    assert engine.tasteAlignmentScore(embeddingEngine, ["", "   "], cooking) == 0.0
    assert 0.0 <= engine.tasteAlignmentScore(embeddingEngine, ["cooking recipes", "travel"], cooking) <= 1.0


def testExposureGovernor():
    governor = engine.ExposureGovernor()
    allowedCount = sum(1 for index in range(200) if governor.allow("u1", "garbagePost", 0.05))
    assert 1 <= allowedCount <= 20
    assert all(governor.allow("u1", "goodPost", 1.0) for index in range(50))
    assert governor.allow("u2", "garbagePost", 1.0)


def buildFixture(roadmap, tastes):
    trendCentroid = engine.computeTrendCentroid(embeddingEngine, ["python tutorial for beginners", "latest phone review", "cooking pasta at home"])
    trendEmbedding = embeddingEngine.embed("trending topics today")
    now = time.time()
    posts = [
        engine.PostRecord("good1", "a1", "a clear step by step python tutorial for beginners with real code", True, now - 3600, False, 0.0, 0.02, 0.45, 500, 80, 10, None, None, ["python basics", "data structures in python"]),
        engine.PostRecord("video1", "a2", "quick phone unboxing video", False, now - 7200, True, 120.0, 0.05, 0.30, 300, 40, 5, None, None, ["phone reviews", "gadget news"]),
        engine.PostRecord("garbage1", "a3", "buy now buy now click click click", False, now - 100, False, 0.0, 0.40, 0.02, 50, 0, 0, None, None, []),
    ]
    events = {
        "good1": [buildEvent("good1", dwellSeconds=40.0, liked=True, sessionSeconds=90.0)],
        "video1": [buildEvent("video1", dwellSeconds=60.0, watchedFraction=0.5, sessionSeconds=90.0)],
        "garbage1": [buildEvent("garbage1", dwellSeconds=0.8, sessionSeconds=90.0)],
    }
    user = buildUser(accessibilityScore=0.7, keyboardConcepts=["python", "code"], dwellHistorySeconds=[10.0, 20.0, 30.0], deviceAgeYears=3.0, deviceIsCharging=True, deviceBandwidthMbps=10.0, roadmap=roadmap, tastes=tastes)
    return user, posts, events, trendCentroid, trendEmbedding, now


def testRankFeedIntegration():
    user, posts, events, centroid, trend, now = buildFixture(None, [])
    results = engine.rankFeedForUser(user, posts, events, embeddingEngine, centroid, trend, now)
    assert len(results) == 3
    scores = [entry.finalScore for entry in results]
    assert scores == sorted(scores, reverse=True)
    for entry in results:
        assert entry.exposureCap in (0.05, 1.0)
        assert entry.finalScore >= 0.0
        assert entry.roadmapScore == 0.0
        assert entry.tasteScore == 0.0


def testRankFeedRoadmapAndTasteBoost():
    user, posts, events, centroid, trend, now = buildFixture("become a python developer", ["python programming"])
    results = engine.rankFeedForUser(user, posts, events, embeddingEngine, centroid, trend, now)
    byId = {entry.postId: entry for entry in results}
    assert byId["good1"].roadmapScore > byId["garbage1"].roadmapScore
    assert byId["good1"].tasteScore > byId["garbage1"].tasteScore
    assert byId["good1"].tasteScore > 0.0


def testTasteWeightZeroDisablesBoost():
    user, posts, events, centroid, trend, now = buildFixture(None, ["python programming"])
    boosted = engine.rankFeedForUser(user, posts, events, embeddingEngine, centroid, trend, now, None, 0.5)
    disabled = engine.rankFeedForUser(user, posts, events, embeddingEngine, centroid, trend, now, None, 0.0)
    boostedGood = [entry for entry in boosted if entry.postId == "good1"][0]
    disabledGood = [entry for entry in disabled if entry.postId == "good1"][0]
    assert boostedGood.finalScore > disabledGood.finalScore


def testRankFeedGovernorRuns():
    user, posts, events, centroid, trend, now = buildFixture(None, [])
    governor = engine.ExposureGovernor()
    results = engine.rankFeedForUser(user, posts, events, embeddingEngine, centroid, trend, now, governor)
    assert len(results) == 3


def testApiRankRoundTrip():
    payload = {
        "user": {"userId": "apiUser", "roadmap": "learn python", "tastes": ["python programming"], "dwellHistorySeconds": [5, 10, 20]},
        "posts": [
            {"postId": "p1", "text": "python tutorial for beginners with examples", "ageHours": 2},
            {"postId": "p2", "text": "grilled fish with lemon and herbs", "ageHours": 2},
        ],
        "events": [{"postId": "p1", "dwellSeconds": 30, "liked": True}],
        "topK": 5,
    }
    response = api.rankJson(service, payload)
    assert response["count"] == 2
    assert response["items"][0]["rank"] == 1
    assert response["items"][0]["postId"] == "p1"
    assert response["latencyMs"] > 0


def testApiValidationRejectsBadInput():
    failed = False
    try:
        api.rankJson(service, {"user": {"userId": "x"}, "posts": []})
    except Exception:
        failed = True
    assert failed
    failed = False
    try:
        api.rankJson(service, {"user": {"userId": "x", "accessibilityScore": 5}, "posts": [{"postId": "p", "text": "t"}]})
    except Exception:
        failed = True
    assert failed


def testApiStatsTrack():
    before = service.stats()["requests"]
    api.rankJson(service, {"user": {"userId": "s"}, "posts": [{"postId": "p", "text": "hello world"}]})
    assert service.stats()["requests"] == before + 1
'''

with open("contentRankingEngine.py", "w") as handle:
    handle.write(engineSource)
with open("rankingApi.py", "w") as handle:
    handle.write(apiSource)
with open("testContentRankingEngine.py", "w") as handle:
    handle.write(testSource)

testRun = subprocess.run([sys.executable, "-m", "pytest", "testContentRankingEngine.py", "-q", "-x", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(testRun.stdout[-3500:])
print(testRun.stderr[-1500:])
if testRun.returncode != 0:
    raise SystemExit("tests failed, fix before launching the app")

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import contentRankingEngine as engine
import rankingApi as api

service = api.RankingService()

exampleFeed = """python tutorial for beginners covering variables loops and functions with runnable code examples | ageHours=2 | author=alice | artifact=true
machine learning explained simply: how a neural network learns from data | ageHours=5 | author=bob
unboxing the newest smartphone with camera tests | ageHours=3 | video=120 | author=carol
creamy garlic pasta ready in fifteen minutes | ageHours=1 | author=dave
python tips for beginners | ageHours=200 | author=alice
buy now click click click free money | ageHours=0.2 | grammar=0.45 | metadata=0.02 | author=spam
beginner strength training routine for home | ageHours=8 | author=erin
budget travel tips for europe with packing list | ageHours=12 | author=frank"""


def parseFeed(feedText):
    posts = []
    for lineIndex, rawLine in enumerate(feedText.strip().splitlines()):
        line = rawLine.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        post = {"postId": "post" + str(lineIndex + 1), "text": parts[0]}
        for option in parts[1:]:
            if "=" not in option:
                continue
            key, value = [piece.strip() for piece in option.split("=", 1)]
            lowered = key.lower()
            if lowered == "agehours":
                post["ageHours"] = float(value)
            elif lowered == "author":
                post["authorId"] = value
            elif lowered == "video":
                post["isVideo"] = True
                post["videoLengthSeconds"] = float(value)
            elif lowered == "artifact":
                post["hasArtifact"] = value.lower() in ("true", "1", "yes")
            elif lowered == "grammar":
                post["grammarPenalty"] = float(value)
            elif lowered == "metadata":
                post["metadataScore"] = float(value)
            elif lowered == "impressions":
                post["impressionCount"] = int(value)
            elif lowered == "id":
                post["postId"] = value
        posts.append(post)
    return posts


def parseEvents(eventText):
    events = []
    for rawLine in eventText.strip().splitlines():
        line = rawLine.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        event = {"postId": parts[0], "dwellSeconds": float(parts[1]) if len(parts) > 1 else 5.0}
        for option in parts[2:]:
            if "=" not in option:
                continue
            key, value = [piece.strip() for piece in option.split("=", 1)]
            lowered = key.lower()
            if lowered == "liked":
                event["liked"] = value.lower() in ("true", "1", "yes")
            elif lowered == "replied":
                event["replied"] = value.lower() in ("true", "1", "yes")
            elif lowered == "visible":
                event["viewportVisibleFraction"] = float(value)
            elif lowered == "watched":
                event["watchedFraction"] = float(value)
            elif lowered == "reply":
                event["replyText"] = value
        events.append(event)
    return events


def splitList(text):
    return [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]


def buildChart(items, posts):
    textById = {post["postId"]: post["text"] for post in posts}
    figure, axes = plt.subplots(1, 2, figsize=(15, max(4, 0.45 * len(items) + 2)))
    labels = [str(item["rank"]) + ". " + textById.get(item["postId"], item["postId"])[:38] for item in items][::-1]
    scores = [item["finalScore"] for item in items][::-1]
    axes[0].barh(labels, scores)
    axes[0].set_title("Final score")
    axes[0].set_xscale("log") if min(scores) > 0 and max(scores) / max(min(scores), 1e-12) > 50 else None
    components = ["engagementQuality", "roadmapScore", "tasteScore", "explorationScore"]
    bottoms = np.zeros(len(items))
    for component in components:
        values = np.array([item[component] for item in items][::-1])
        axes[1].barh(labels, values, left=bottoms, label=component)
        bottoms += values
    axes[1].set_title("Component breakdown")
    axes[1].legend(fontsize=8)
    axes[1].set_yticklabels([])
    plt.tight_layout()
    return figure


def runRanking(feedText, eventText, userId, roadmap, tastes, concepts, dwellText, accessibility, deviceAge, charging, bandwidth, tasteWeight, roadmapWeight, topK, enforceCap, trendText, latitude, longitude):
    try:
        posts = parseFeed(feedText)
        if not posts:
            return "Paste at least one post.", None, None, ""
        dwellHistory = [float(value) for value in splitList(dwellText)]
        payload = {
            "user": {
                "userId": userId or "anonymous",
                "accessibilityScore": accessibility,
                "gpsLatitude": latitude if latitude not in (None, "") else None,
                "gpsLongitude": longitude if longitude not in (None, "") else None,
                "keyboardConcepts": splitList(concepts),
                "dwellHistorySeconds": dwellHistory,
                "deviceAgeYears": deviceAge,
                "deviceIsCharging": bool(charging),
                "deviceBandwidthMbps": bandwidth,
                "roadmap": roadmap.strip() if roadmap and roadmap.strip() else None,
                "tastes": tastes or [],
            },
            "posts": posts,
            "events": parseEvents(eventText),
            "trendingTexts": splitList(trendText),
            "tasteWeight": tasteWeight,
            "roadmapWeight": roadmapWeight,
            "topK": int(topK),
            "enforceExposureCap": bool(enforceCap),
        }
        response = api.rankJson(service, payload)
        textById = {post["postId"]: post["text"] for post in posts}
        rows = []
        for item in response["items"]:
            rows.append({
                "rank": item["rank"],
                "post": textById.get(item["postId"], item["postId"])[:90],
                "finalScore": round(item["finalScore"], 6),
                "quality": round(item["engagementQuality"], 4),
                "regency": round(item["regencyScore"], 4),
                "mastery": round(item["masteryScore"], 4),
                "roadmap": round(item["roadmapScore"], 4),
                "taste": round(item["tasteScore"], 4),
                "explore": round(item["explorationScore"], 4),
                "exposureCap": item["exposureCap"],
            })
        frame = pd.DataFrame(rows)
        summary = "ranked " + str(response["count"]) + " posts in " + format(response["latencyMs"], ".1f") + " ms | service stats: " + json.dumps(service.stats())
        return summary, frame, buildChart(response["items"], posts), json.dumps(response, indent=2)
    except Exception as error:
        return "error: " + str(error), None, None, ""


tasteChoices = ["python programming", "machine learning", "cooking recipes", "smartphones and gadgets", "fitness and workouts", "travel and adventure", "photography", "personal finance", "music", "gaming", "startups and business", "science and space"]

with gr.Blocks(title="Feed Ranking Engine", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Feed Ranking Engine\nPaste posts, set the user's taste and context, get a ranked feed. Real sentence embeddings, engagement, recency, roadmap and taste scoring.")
    with gr.Row():
        with gr.Column(scale=3):
            feedBox = gr.Textbox(label="Posts, one per line: text | ageHours=2 | author=name | video=120 | artifact=true | grammar=0.1 | metadata=0.3 | impressions=500", lines=12, value=exampleFeed)
            eventBox = gr.Textbox(label="Scroller events, optional, one per line: postId | dwellSeconds | liked=true | replied=true | watched=0.5 | reply=text", lines=5, value="post1 | 40 | liked=true\npost2 | 18\npost3 | 60 | watched=0.5\npost6 | 0.8 | reply=never again, worst post")
        with gr.Column(scale=2):
            userBox = gr.Textbox(label="User id", value="user1")
            roadmapBox = gr.Textbox(label="Roadmap (plain string)", value="become a python developer and learn machine learning")
            tasteBox = gr.CheckboxGroup(label="Taste (pick any)", choices=tasteChoices, value=["python programming", "machine learning"])
            conceptBox = gr.Textbox(label="Keyboard concepts, comma separated", value="python, code, learning")
            dwellBox = gr.Textbox(label="Dwell history in seconds, comma separated", value="8, 15, 25, 40, 12")
            trendBox = gr.Textbox(label="Trending texts, comma separated, optional", value="python tutorial for beginners, machine learning explained simply, latest smartphone review")
    with gr.Row():
        accessibilitySlider = gr.Slider(0, 1, value=0.7, step=0.05, label="Accessibility score")
        deviceAgeSlider = gr.Slider(0, 15, value=3, step=0.5, label="Device age in years")
        bandwidthSlider = gr.Slider(0.1, 100, value=12, step=0.1, label="Bandwidth Mbps")
        chargingBox = gr.Checkbox(label="Device charging", value=True)
    with gr.Row():
        tasteWeightSlider = gr.Slider(0, 2, value=engine.tasteWeightDefault, step=0.05, label="Taste weight")
        roadmapWeightSlider = gr.Slider(0, 2, value=engine.roadmapWeight, step=0.05, label="Roadmap weight")
        topKSlider = gr.Slider(1, 200, value=20, step=1, label="Top K")
        capBox = gr.Checkbox(label="Enforce 5 percent low quality exposure cap", value=True)
    with gr.Row():
        latitudeBox = gr.Number(label="Latitude, optional (used lightly)", value=None)
        longitudeBox = gr.Number(label="Longitude, optional (used lightly)", value=None)
    runButton = gr.Button("Rank feed", variant="primary")
    summaryBox = gr.Textbox(label="Summary")
    tableBox = gr.Dataframe(label="Ranked feed")
    plotBox = gr.Plot(label="Scores")
    jsonBox = gr.Code(label="API response JSON", language="json")
    runButton.click(
        runRanking,
        inputs=[feedBox, eventBox, userBox, roadmapBox, tasteBox, conceptBox, dwellBox, accessibilitySlider, deviceAgeSlider, chargingBox, bandwidthSlider, tasteWeightSlider, roadmapWeightSlider, topKSlider, capBox, trendBox, latitudeBox, longitudeBox],
        outputs=[summaryBox, tableBox, plotBox, jsonBox],
        api_name="rank",
    )

def benchmarkRun():
    rng = random.Random(3)
    bank = ["python tutorial for beginners", "machine learning explained simply", "creamy garlic pasta recipe", "smartphone camera comparison", "beginner strength training", "budget travel in europe", "buy now click click free money"]
    sizes = [10, 25, 50, 100, 200, 400]
    rows = []
    for size in sizes:
        posts = [{"postId": "b" + str(index), "text": rng.choice(bank) + " variant " + str(rng.randint(0, 40)), "ageHours": rng.uniform(0.1, 200)} for index in range(size)]
        payload = {"user": {"userId": "bench", "roadmap": "learn python and machine learning", "tastes": ["python programming"], "dwellHistorySeconds": [5, 10, 20, 40]}, "posts": posts, "events": [{"postId": "b0", "dwellSeconds": 20}], "topK": size}
        api.rankJson(service, payload)
        timings = []
        for repeat in range(4):
            started = __import__("time").perf_counter()
            api.rankJson(service, payload)
            timings.append((__import__("time").perf_counter() - started) * 1000)
        rows.append({"posts": size, "meanMs": float(np.mean(timings)), "stdMs": float(np.std(timings)), "postsPerSecond": size / (np.mean(timings) / 1000)})
    return pd.DataFrame(rows)


benchmarkFrame = benchmarkRun()
print(benchmarkFrame.round(2).to_string(index=False))

figure, axes = plt.subplots(1, 3, figsize=(19, 5))
axes[0].errorbar(benchmarkFrame["posts"], benchmarkFrame["meanMs"], yerr=benchmarkFrame["stdMs"], marker="o", capsize=3)
axes[0].set_title("API latency vs feed size")
axes[0].set_xlabel("posts")
axes[0].set_ylabel("ms")
axes[0].grid(alpha=0.3)
axes[1].bar([str(value) for value in benchmarkFrame["posts"]], benchmarkFrame["postsPerSecond"])
axes[1].set_title("Throughput")
axes[1].set_xlabel("posts")
axes[1].set_ylabel("posts per second")
axes[1].grid(alpha=0.3, axis="y")
anchors = [(0.05, 0.01), (0.25, 0.20), (0.50, 0.50), (0.75, 0.70), (1.00, 0.10)]
fractions = np.linspace(0, 1, 200)
axes[2].plot(fractions, [engine.piecewiseInterpolate(value, anchors) for value in fractions])
axes[2].scatter([anchor[0] for anchor in anchors], [anchor[1] for anchor in anchors], color="red", zorder=3)
axes[2].set_title("Video watch score curve")
axes[2].set_xlabel("fraction watched")
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()

demoResponse = api.rankJson(service, {"user": {"userId": "demo", "roadmap": "become a python developer", "tastes": ["python programming"]}, "posts": parseFeed(exampleFeed), "topK": 5})
print(json.dumps(demoResponse, indent=2)[:1500])

app.launch(share=True, debug=False)
