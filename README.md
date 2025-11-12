# Week 7: Autoscaling and Observability for Iris Classification API on GKE

This project extends the Week 5 CI/CD pipeline by deploying the Iris Classification FastAPI model to **Google Kubernetes Engine (GKE)** and enabling **horizontal pod autoscaling (HPA)**, logging, and performance observability.

---

## 🚀 Objectives
1. Extend the previous CI/CD pipeline to support **scalable deployments**.
2. Deploy the FastAPI model on GKE with **Docker + Artifact Registry**.
3. Implement **logging, exception handling, probes**, and **autoscaling**.
4. Perform **stress testing** using `wrk` to simulate concurrent inferences.
5. Analyze system behavior, scaling, and bottlenecks under load.

---

## 🧱 Architecture Overview
- FastAPI app served via Uvicorn  
- Containerized using Docker  
- Deployed on GKE with HPA and load balancing  
- Benchmarked using `wrk`  

**Public Endpoint Example:**


---

## 📊 Performance & Autoscaling Summary

| Scenario | Connections | Avg Latency | Requests/sec | Errors | Autoscaling Behavior |
|-----------|--------------|--------------|---------------|--------|----------------------|
| Baseline | 100 | 188 ms | 529 | 0 | 1 pod |
| Stress Test | 1000 | 440 ms | 1688 | ~1140 | 1→3 pods |
| Bottleneck | 2000 | 1000+ ms | ~800 | High | Fixed 1 pod |

---

## 🏁 Conclusion
This week’s work demonstrates how to containerize, deploy, and scale an ML inference API using GCP-native tooling.  
Autoscaling proved essential to maintain performance under load, and observability helped visualize system health in real-time.

---

**Author:** *Anupama Tiwari*  
*Week 6 – GKE Autoscaling & Observability Assignment*
