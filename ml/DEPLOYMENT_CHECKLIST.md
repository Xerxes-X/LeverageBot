# Production Deployment Checklist

Complete checklist before deploying ML service to production.

---

## Pre-Deployment

### Model Training & Validation

- [ ] Trained on sufficient data (minimum 60 days, recommended 90+ days)
- [ ] Out-of-sample validation completed
- [ ] Performance meets targets:
  - [ ] Win rate: 58-62%
  - [ ] Sharpe ratio: 1.5-2.0
  - [ ] Inference latency: <10ms (p95)
  - [ ] Max drawdown: <15%
- [ ] Backtesting completed with realistic transaction costs
- [ ] Model tested across different market conditions (trending, ranging, volatile)

### Code Quality

- [ ] All tests passing: `pytest tests/ -v`
- [ ] No critical security issues
- [ ] API documentation up-to-date
- [ ] Code reviewed by team member
- [ ] Logging properly configured

### Infrastructure

- [ ] Docker image built and tested
- [ ] Health check endpoint verified
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Log aggregation set up
- [ ] Backup strategy for models

---

## Security

### API Security

- [ ] CORS properly configured (not `allow_origins=["*"]` in production)
- [ ] Rate limiting implemented
- [ ] Authentication/authorization (if needed)
- [ ] Input validation on all endpoints
- [ ] Error messages don't leak sensitive info

### Model Security

- [ ] Model files stored securely (read-only access)
- [ ] Model versioning implemented
- [ ] Rollback plan for bad models
- [ ] Model files not committed to git (use .gitignore)

### Environment

- [ ] Secrets managed properly (not hardcoded)
- [ ] Environment variables documented
- [ ] Production config separate from dev config
- [ ] Access logs enabled

---

## Performance

### Latency Optimization

- [ ] Feature caching implemented
- [ ] Model loaded at startup (not per-request)
- [ ] Database connections pooled (if using DB)
- [ ] Unnecessary logging removed from hot paths
- [ ] Load tested with expected traffic

### Resource Allocation

- [ ] CPU allocation: 4+ cores recommended
- [ ] Memory allocation: 4GB+ recommended
- [ ] Disk space: 50GB+ for logs and models
- [ ] Network bandwidth sufficient

### Monitoring

- [ ] Latency metrics collected (p50, p95, p99)
- [ ] Prediction count tracked
- [ ] Error rate monitored
- [ ] Resource utilization monitored (CPU, memory)
- [ ] Alerts configured for:
  - [ ] High latency (>20ms)
  - [ ] High error rate (>5%)
  - [ ] Service down
  - [ ] Model drift detected

---

## Integration

### Rust Bot Integration

- [ ] ML client added to bot codebase
- [ ] Configuration tested
- [ ] Fallback behavior verified (bot works if ML service fails)
- [ ] End-to-end integration tested
- [ ] Dry-run mode tested first

### Service Communication

- [ ] Network connectivity verified
- [ ] Timeouts properly configured
- [ ] Retry logic implemented
- [ ] Circuit breaker (optional but recommended)
- [ ] Health check polling from bot

---

## Deployment Process

### Pre-Deployment

- [ ] Deployment plan documented
- [ ] Rollback plan documented
- [ ] Maintenance window scheduled (if needed)
- [ ] Team notified
- [ ] Backup taken

### Deployment Steps

1. [ ] Stop old ML service (if running)
2. [ ] Deploy new ML service container
3. [ ] Verify health check passes
4. [ ] Test prediction endpoint manually
5. [ ] Monitor logs for errors
6. [ ] Restart bot to connect to new ML service
7. [ ] Verify bot receives ML predictions
8. [ ] Monitor for 30 minutes

### Post-Deployment

- [ ] Verify metrics dashboard
- [ ] Check latency is within target (<10ms p95)
- [ ] Verify prediction accuracy matches backtests
- [ ] Monitor error logs
- [ ] Document deployment (date, version, changes)

---

## Monitoring & Maintenance

### Daily Checks

- [ ] Service uptime: >99.9%
- [ ] Average latency: <10ms
- [ ] Error rate: <1%
- [ ] No critical errors in logs

### Weekly Checks

- [ ] Performance metrics vs. targets
- [ ] Model drift analysis
- [ ] Feature importance stability
- [ ] Resource utilization trends

### Monthly Checks

- [ ] Retrain model with new data
- [ ] Backtest new model
- [ ] Compare new vs. old model performance
- [ ] Update model if improvement >5%
- [ ] Review and optimize configuration

---

## Rollback Triggers

Immediately rollback if:

- [ ] Win rate drops below 52% for 3+ days
- [ ] Sharpe ratio drops below 1.0
- [ ] Latency exceeds 20ms (p95) consistently
- [ ] Error rate exceeds 5%
- [ ] Service crashes repeatedly
- [ ] Bot reports ML service unreachable

**Rollback Procedure:**
1. Stop current ML service
2. Deploy previous known-good version
3. Verify health check
4. Restart bot
5. Monitor for 30 minutes
6. Investigate root cause

---

## Disaster Recovery

### Model Corruption

- [ ] Keep last 3 model versions
- [ ] Automated backup before each deployment
- [ ] Tested restore procedure
- [ ] Recovery time objective: <5 minutes

### Service Outage

- [ ] Bot continues with fallback signals
- [ ] Alerts trigger immediately
- [ ] On-call engineer notified
- [ ] Recovery time objective: <15 minutes

### Data Loss

- [ ] Training data backed up regularly
- [ ] Model artifacts backed up
- [ ] Configuration backed up
- [ ] Recovery time objective: <1 hour

---

## Documentation

- [ ] API documentation complete (OpenAPI/Swagger)
- [ ] Integration guide updated
- [ ] Runbook created for common issues
- [ ] Architecture diagram updated
- [ ] Team training completed

---

## Sign-Off

- [ ] Development team lead approval
- [ ] DevOps/Infrastructure approval
- [ ] Security review completed
- [ ] Business stakeholder notified

---

**Deployment Date:** __________

**Deployed By:** __________

**Model Version:** __________

**Rollback Plan Verified:** [ ] Yes [ ] No

**Notes:**
