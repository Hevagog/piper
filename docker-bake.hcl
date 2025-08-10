group "default" {
  targets = ["base", "luigi-scheduler", "luigi-worker", "orchestrator","nn-worker"]
}
group "base" {
  targets = ["base", "luigi-scheduler", "luigi-worker", "orchestrator"]
}

group "rust-nn" {
  targets = ["nn-worker"]
}

target "base" {
  context = "."
  dockerfile = "docker/base.Dockerfile"
  tags = ["piper-base:latest"]
}

target "luigi-scheduler" {
  context = "."
  dockerfile = "docker/luigi-scheduler.Dockerfile"
  tags = ["piper-scheduler:latest"]
  depends_on = ["base"]
}

target "luigi-worker" {
  context = "."
  dockerfile = "docker/luigi-worker.Dockerfile"
  tags = ["piper-worker:latest"]
  depends_on = ["base", "luigi-scheduler"]
}

target "orchestrator" {
  context = "."
  dockerfile = "docker/orchestrator.Dockerfile"
  tags = ["piper-orchestrator:latest"]
  depends_on = ["base", "luigi-scheduler"]
}

target "nn-worker" {
  context = "."
  dockerfile = "docker/nn-worker.Dockerfile"
  tags = ["piper-nn-worker:latest"]
  depends_on = ["base"]
}