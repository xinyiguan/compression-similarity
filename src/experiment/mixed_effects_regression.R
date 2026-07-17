# install.packages(c("lmerTest", "MuMIn"))
library(lmerTest)
library(MuMIn)


data_ctx_false <- read.csv("../../data/result/(ctx=False)/longform_pairwise_similarity.csv")
data_ctx_false$D_ppm_norm <- (data_ctx_false$D_ppm - mean(data_ctx_false$D_ppm)) / sd(data_ctx_false$D_ppm)
data_ctx_false$D_stitch_norm <- (data_ctx_false$D_stitch - mean(data_ctx_false$D_stitch)) / sd(data_ctx_false$D_stitch)


data_ctx_true <- read.csv("../../data/result/(ctx=True)/longform_pairwise_similarity.csv")
data_ctx_true$D_ppm_norm <- (data_ctx_true$D_ppm - mean(data_ctx_true$D_ppm)) / sd(data_ctx_true$D_ppm)
data_ctx_true$D_stitch_norm <- (data_ctx_true$D_stitch - mean(data_ctx_true$D_stitch)) / sd(data_ctx_true$D_stitch)


# ctx = False
model_ctx_false <- lmer(
  D_ppm_norm ~ D_stitch_norm + (D_stitch_norm|piece_i) + (D_stitch_norm|piece_j),
  data = data_ctx_false
)
summary(model_ctx_false)
r.squaredGLMM(model_ctx_false)


# ctx = True
model_ctx_true <- lmer(
  D_ppm_norm ~ D_stitch_norm + (D_stitch_norm|piece_i) + (D_stitch_norm|piece_j),
  data = data_ctx_true
)
summary(model_ctx_true)
r.squaredGLMM(model_ctx_true)

