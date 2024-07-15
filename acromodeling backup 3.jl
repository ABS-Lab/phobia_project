### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ ffdbec30-2bd7-11ee-2284-1bb2a6d11973
using CSV, DataFrames, Plots, Random, Statistics, GLM, StatsPlots, Tables, Printf, MixedModels, Pipe, NaNStatistics

# ╔═╡ 27ba71bf-1f2e-4a79-89c7-2ccdbf3ec6f8
md"# Modeling Notebook"

# ╔═╡ bf0aca2a-99d2-4ef0-a1e4-aa4e48e265fc
cond_num = 1

# ╔═╡ 8d099526-be2e-4543-bece-73721bdce561
dfpre = CSV.File(raw"E:\+WORKSPACE\acro\affvids_physio_with_phobia.csv") |> DataFrame;

# ╔═╡ 013bfd5b-8c78-4f6b-8d6b-fd83ace08482
dfpre[:, :participant_num_str] = dfpre[:, :participant_num] .|> string

# ╔═╡ f16dac04-f3a5-48f6-8eb1-46dd0abe16d4
mean_miss(x) = mean(skipmissing(x))

# ╔═╡ 2d64c863-ed4f-466d-bcf7-882397d7bf5e
std_miss(x) = std(skipmissing(x))

# ╔═╡ 8b8de4be-9f00-4a86-8151-1746d307f40f
df1pre = dfpre[dfpre[:, :video_condition] .== cond_num, :]

# ╔═╡ 313f2cd7-9dd6-462c-83e6-86232e8dfa9e
df2pre_pre = @pipe dfpre |> 
	groupby(_,:participant_num) |> 
	transform(_, [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> mean_miss, [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> std_miss) |> 
	transform(_, [:video_scr, :video_scr_mean_miss]=> ByRow(-) => :video_scr_cmc, [:video_hp, :video_hp_mean_miss]=> ByRow(-) => :video_hp_cmc, [:resp_arousal, :resp_arousal_mean_miss]=> ByRow(-) => :resp_arousal_cmc, [:resp_fear, :resp_fear_mean_miss]=> ByRow(-) => :resp_fear_cmc, [:resp_valence, :resp_valence_mean_miss]=> ByRow(-) => :resp_valence_cmc) |>
	transform(_, [:video_scr_cmc, :video_scr_std_miss]=> ByRow(/) => :video_scr_z, [:video_hp_cmc, :video_hp_std_miss]=> ByRow(/) => :video_hp_z, [:resp_arousal_cmc, :resp_arousal_std_miss]=> ByRow(/) => :resp_arousal_z, [:resp_fear_cmc, :resp_fear_std_miss]=> ByRow(/) => :resp_fear_z,
	[:resp_valence_cmc, :resp_valence_std_miss]=> ByRow(/) => :resp_valence_z)

# ╔═╡ 2d0f1009-7f92-4f91-bc26-2a7cd5b4ec5e
df2pre = df2pre_pre[df2pre_pre[:, :video_condition] .== cond_num, :]

# ╔═╡ 277630ab-e88d-43bf-a511-a33ab23cddb1
md"---"

# ╔═╡ 9a2edcaa-93d4-44d0-909b-b4f6e3f548f7
md"## Correlation Table"

# ╔═╡ 212a249e-6b19-4ddf-9125-93edd3b36977
df3pre1 = @pipe dfpre[dfpre[:, :video_condition] .== 1, :] |>
	groupby(_,:participant_num) |>
	combine(_, [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> mean_miss)[:,2:end]

# ╔═╡ c214f257-a167-4dd5-bc20-41386c398001
df3vars1 = rename(df3pre1, names(df3pre1) .|> (x -> string("heights_", x)))

# ╔═╡ 0201632a-e3e0-447d-a6a8-5965e1570a64
df3pre2 = @pipe dfpre[dfpre[:, :video_condition] .== 2, :] |>
	groupby(_,:participant_num) |>
	combine(_, [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> mean_miss)[:,2:end]

# ╔═╡ 5e5da2db-5aac-4296-abad-4535ff669e8c
df3vars2 = rename(df3pre2, names(df3pre2) .|> (x -> string("social_", x)))

# ╔═╡ e9eab7cf-e11f-45a3-a85b-4fde496cfb2d
df3pre3 = @pipe dfpre[dfpre[:, :video_condition] .== 3, :] |>
	groupby(_,:participant_num) |>
	combine(_, [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> mean_miss)[:,2:end]

# ╔═╡ 126b8f45-ae49-4bc9-bf6e-c2425d864ce2
df3vars3 = rename(df3pre3, names(df3pre3) .|> (x -> string("spider_", x)))

# ╔═╡ aec29b59-5cf5-4b2d-9853-12f5e7350f2c
df3 = @pipe dfpre |> 
	groupby(_,:participant_num) |> 
	combine(_, [:heights_phobia, :social_phobia, :spider_phobia] .=> mean_miss) 
# , [:video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> std_miss

# ╔═╡ d4079ef4-5441-4072-a055-fc372b89843f
df3_full = reduce(hcat, [df3, df3vars1, df3vars2, df3vars3])[:,2:end]

# ╔═╡ 229d3ba7-9a67-489f-9043-ccf1134d9de1
corcolnames = names(df3_full) .|> (x -> x[1:end-10])

# ╔═╡ e00e776d-3f7d-43d8-8da8-d58419234802
cordf = DataFrame(nancor(Matrix(df3_full)), corcolnames)

# ╔═╡ 7964d5b3-c711-4feb-948f-f55853b4e33a
df3_extra = @pipe dfpre |> 
	groupby(_,:participant_num) |> 
	combine(_, [:heights_phobia, :video_scr, :video_hp, :resp_arousal, :resp_fear, :resp_valence] .=> mean_miss) 

# ╔═╡ c6bd9ec5-dc72-4e3e-a843-6ef485bb614c
corcolnamesextra = names(df3_extra[:,2:end]) .|> (x -> x[1:end-10])

# ╔═╡ 515c6bc3-a59b-4b71-acae-e639e1a9bd2a
cordfextra = DataFrame(nancor(Matrix(df3_extra[:,2:end])), corcolnamesextra)

# ╔═╡ 53807a24-0fd1-44c3-9808-4f74fa2d3fe9
md"# Test Area"

# ╔═╡ 63522390-0eea-4ccc-9215-5a54c1642511
df1pre 

# ╔═╡ 8f813e05-4dc3-48e4-af79-bafe2b643338
collist = ["video",
"participant_num",
"resp_exp_fear",
"rt_exp_fear",
"resp_current_anxiety",
"rt_current_anxiety",
"resp_fear",
"resp_anxiety",
"resp_arousal",
"resp_valence",
"rt_fear",
"rt_anxiety",
"rt_arousal",
"rt_valence",
"video_hp",
"video_scr",
"base_ECG",
"base_scr",
"hp_change_video",
"scr_change_video",
"heights_phobia",
"participant_num_str"];

# ╔═╡ d5a188fb-8fc4-458d-a998-c89593d87c38
dfcleancol = coalesce.(df1pre[:, collist], 0)

# ╔═╡ d940c157-8556-4d4a-92cc-380b3a7942b7
collist_df2 = ["video",
"participant_num",
"resp_exp_fear",
"rt_exp_fear",
"resp_current_anxiety",
"rt_current_anxiety",
"resp_fear",
"resp_anxiety",
"resp_arousal",
"resp_valence",
"rt_fear",
"rt_anxiety",
"rt_arousal",
"rt_valence",
"video_hp",
"video_scr",
"base_ECG",
"base_scr",
"hp_change_video",
"scr_change_video",
"heights_phobia",
"participant_num_str",
"video_scr_mean",
"resp_arousal_mean",
"resp_fear_mean",
"resp_valence_mean",
"video_scr_std",
"resp_arousal_std",
"resp_fear_std",
"resp_valence_std",
"video_scr_cmc",
"resp_arousal_cmc",
"resp_fear_cmc",
"resp_valence_cmc",
"video_scr_z",
"resp_arousal_z",
"resp_fear_z",
"resp_valence_z"];

# ╔═╡ 323785ee-1332-47a5-8726-f25aa002ff81
df2cleancol = coalesce.(df2pre[:, collist_df2], 0)

# ╔═╡ 409c55aa-a290-4403-a707-42e6b39c1943
dfcleancol[dfcleancol[:,:participant_num] .== 224,:]

# ╔═╡ 6be6b7ec-f21d-4fa2-b99b-761bdc55c8ee
dfcleancol[:,"video_scr"]

# ╔═╡ b0263688-0f15-47b7-bf66-299917962941
function comparevar(var1) #with phobia correlation
	return cor(dfcleancol[:,"heights_phobia"],dfcleancol[:,var1])
end

# ╔═╡ 70c5c4cc-087a-4f24-a2ff-3b1349e969b1
comparevar("video_scr")

# ╔═╡ 9c98b6f4-9c04-44e3-ad4a-539ce588f9f4
comparevar("resp_arousal")

# ╔═╡ dc357f8d-1960-476e-a728-4975f9ec4444
comparevar("resp_fear")

# ╔═╡ c80d8246-19c2-4c63-8301-861e52c6f6ed
plotdata = [comparevar("video_scr"), comparevar("resp_arousal"), comparevar("resp_fear")];

# ╔═╡ 3bfd7f99-f2a5-4d02-a373-7d022ed1f877
plotcols = ["video_scr", "resp_arousal", "resp_fear"];

# ╔═╡ 9be1cff9-3cfd-4299-976b-8adeeeb5b54f
plot(bar(plotcols, plotdata))

# ╔═╡ 04cad377-0e86-4baa-bf7c-0a738fa973df
var1 = "video_scr"

# ╔═╡ 5dd5cbf6-df2e-414a-8e24-0d767eaf1af7
var2 = "resp_arousal"

# ╔═╡ b18892c8-3ba5-4558-af64-9c915dff878a
lm1 = lm(@formula(video_scr~ resp_arousal),dfcleancol)

# ╔═╡ e1489517-53a0-4b33-bcfe-480173139711
coef(lm1)[1]

# ╔═╡ d6016ec7-8018-4916-86bc-64d1598f8d06
gdf = groupby(dfcleancol, :participant_num_str)

# ╔═╡ 21b79764-fa90-4fe9-924f-80c22919a90f
gdf[6]

# ╔═╡ e2cc8c45-dc01-44e6-b2d9-c180c5a9cd58
collist2 = [
"resp_exp_fear",
"rt_exp_fear",
"resp_current_anxiety",
"rt_current_anxiety",
"resp_fear",
"resp_anxiety",
"resp_arousal",
"resp_valence",
"rt_fear",
"rt_anxiety",
"rt_arousal",
"rt_valence",
"video_hp",
"video_scr",
"base_ECG",
"base_scr",
"hp_change_video",
"scr_change_video",
"heights_phobia"];

# ╔═╡ 39af73ef-b8de-4af5-9b67-f3f9581384ed
function meantest(series1, series2)
	if sum(series1) == 0 && mean(series1) == 0
		return missing
	end
	dfloc = DataFrame(a=series1, b=series2)
	lm1 = lm(@formula(a~ b),dfloc)
	return (coef(lm1)[1], coef(lm1)[2], loglikelihood(lm1))
end

# ╔═╡ d699231e-48b1-40f7-abdb-a1f5c7954c0d
md"### Fitting SCR to Arousal"

# ╔═╡ d5dd7ff9-18b2-44ed-ade2-cc9dac386d44
df2_subjlvl = combine(gdf, [:video_scr, :resp_arousal] => meantest, :heights_phobia => mean, renamecols=false) |> dropmissing

# ╔═╡ 4d81be85-1611-41d6-907b-dcf411143162
scatter(df2_subjlvl.video_scr_resp_arousal .|> x -> x[1], df2_subjlvl.heights_phobia, label="Intercept")

# ╔═╡ 412538a0-2bb3-4e5d-a78c-65dfdcbbaa08
scatter(df2_subjlvl.video_scr_resp_arousal .|> x -> x[2], df2_subjlvl.heights_phobia, label="Weight")

# ╔═╡ 19195589-73d3-48ae-9553-0c5ebaf0effe
scr_to_aro_int = cor(df2_subjlvl.video_scr_resp_arousal .|> x -> x[1], df2_subjlvl.heights_phobia)

# ╔═╡ 2df570fc-0f9e-4166-82d8-83fcdb115d2f
scr_to_aro_coef = cor(df2_subjlvl.video_scr_resp_arousal .|> x -> x[2], df2_subjlvl.heights_phobia)

# ╔═╡ 4815493b-e5d4-4572-aa07-380da5a2c36d
df2_subjlvl.video_scr_resp_arousal .|> x -> x[3]

# ╔═╡ 39534e93-a4b4-4c28-a7a2-deede208775a
md"### Fitting Arousal to Fear Response"

# ╔═╡ eae03171-3511-4ac5-8065-0b5090159e80
df2_subjlvl_2 = combine(gdf, [:resp_arousal, :resp_fear] => meantest, :heights_phobia => mean, renamecols=false) |> dropmissing

# ╔═╡ d0aefb0a-7ae2-43f7-a0ea-fdf5662c2ca1
scatter(df2_subjlvl_2.resp_arousal_resp_fear .|> x -> x[1], df2_subjlvl.heights_phobia, label="Intercept")

# ╔═╡ 39f10fb6-0383-4742-8ebf-fd0969aeda3c
scatter(df2_subjlvl_2.resp_arousal_resp_fear .|> x -> x[2], df2_subjlvl_2.heights_phobia, label="Weight")

# ╔═╡ 0c5fef51-9132-4a37-bfc3-ff042a82acc5
aro_to_fear_int = cor(df2_subjlvl_2.resp_arousal_resp_fear .|> x -> x[1], df2_subjlvl_2.heights_phobia)

# ╔═╡ 1fa45402-b95c-4cbb-9eb3-dc1e3ec169a8
aro_to_fear_coef = cor(df2_subjlvl_2.resp_arousal_resp_fear .|> x -> x[2], df2_subjlvl_2.heights_phobia)

# ╔═╡ 7dcb34da-5d5e-4389-9757-e5ea1ff38263
plotdata2 = [scr_to_aro_int,scr_to_aro_coef, aro_to_fear_int, aro_to_fear_coef];

# ╔═╡ a4186efc-8752-4bbb-86ed-0160a1ed9359
plotcols2 = ["scr_to_aro_int","scr_to_aro_coef", "aro_to_fear_int", "aro_to_fear_coef"];

# ╔═╡ 299a56a3-5b2f-486e-afdc-2e8adac87962
plot(bar(plotcols2, plotdata2))

# ╔═╡ cc7e0d56-68a6-44e0-b227-784d36abc89f
md"# Test Area 2"

# ╔═╡ 1d3ce898-73a8-4b17-a758-bceaf5d5bb05
df2_subjlvl_hp = combine(gdf, :heights_phobia => mean, renamecols=false);

# ╔═╡ 29bd56bb-757b-40d1-ad46-9874418974d5
dfcleancol

# ╔═╡ 3b9e3928-fef5-446a-b0dc-ba2371a2720a
md"```
begin
## Fitting MLMs
mm2 = fit(LinearMixedModel, @formula(happy ~ gm_GNP + (1|country)), data_happy)
push!(model_fit, aic(mm2))
# extract log likelihood
loglikelihood(mm2)
# extract Akaike's Information Criterion
aic(mm2)
# extract Bayesian Information Criterion
bic(mm2)
# extract degrees of freedom
dof(mm2)
# extract coefficient
coef(mm2)
# extract fixed effects
fixef(mm2)
vcov(mm2)
stderror(mm2)
# extract coefficients table
coeftable(mm2)
# extract variance components
VarCorr(mm2)
# return sigma^2
varest(mm2)
# return tau
VarCorr(mm2).σρ[1][1][1]
# return elements in the components
dump(VarCorr(mm1))
# return sigma
sdest(mm2)
# extract random effects
ranef(mm2)
end
```";

# ╔═╡ 85655803-cef4-4a6f-a9c6-3daf41e93ecb
mm2 = fit(LinearMixedModel, @formula(resp_arousal ~ video_scr + (1|participant_num_str)), dfcleancol)

# ╔═╡ 46020a9f-a305-43f5-87e3-c5f827da27b5
VarCorr(mm2)

# ╔═╡ 1cf2c58b-22db-4271-bb4e-cd93f99058b1
ranef(mm2)

# ╔═╡ b1362f6d-d74c-4ee6-b5a3-7097e9600fc5
var(only(ranef(mm2)))

# ╔═╡ 1dc2d037-0b5a-46d9-97bd-cc220d022411
DataFrame(only(raneftables(mm2)))

# ╔═╡ 9cd7c5e9-b8f2-4cd8-a3b5-5dc632231a64
md"## Mixed Models"

# ╔═╡ aa21b5ee-e94f-4063-b133-3dccee4bab2c
dfcleancol2 = @pipe dfcleancol |> 
            groupby(_,:participant_num_str) |>
            transform(_, [:video_scr, :video_hp, :resp_arousal, :scr_change_video, :resp_valence] .=> mean)|> 
            transform(_, [:video_scr, :video_scr_mean] => ByRow(-) => :video_scr_cmc,
				[:video_hp, :video_hp_mean] => ByRow(-) => :video_hp_cmc,[:resp_arousal, :resp_arousal_mean] => ByRow(-) => :resp_arousal_cmc, 
				[:scr_change_video, :scr_change_video_mean] => ByRow(-) => :scr_change_video_cmc,
				[:resp_valence, :resp_valence_mean] => ByRow(-) => :resp_valence_cmc)


# ╔═╡ 3789cddf-a693-4114-880a-76c531947086
get_trunc(flt) = @sprintf("%.3f", flt)

# ╔═╡ 51cb82d5-6ad3-455d-972b-4b914a3dcdc7
histogram(dfcleancol2.resp_fear)

# ╔═╡ 7bd4ace3-c5f8-4cd4-bd92-d5ec73da2d40
histogram(dfcleancol2.resp_arousal)

# ╔═╡ dc95e5cf-3eb2-4127-8730-e3444568639a
md"## Model 1: src to arousal"

# ╔═╡ 69a13bd2-4154-4897-a964-b7eee3998b24
mm3 = fit(LinearMixedModel, @formula(resp_arousal ~ video_scr_cmc + video_scr_mean + (video_scr_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 97b01db8-8d58-4ab9-9f20-634507db6f97
md"#### Scr to arousal, loglike: $(loglikelihood(mm3))"

# ╔═╡ 13b0dee5-7fb2-4196-9342-4771edf287ac
md"#### Scr to arousal, AIC: $(aic(mm3))"

# ╔═╡ 1da4269e-ee2d-49b8-b537-248f912940cf
mm3coef_i = cor(ranef(mm3)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 0822037d-4284-4e00-8a35-9c2de8f6f4e3
mm3coef_w = cor(ranef(mm3)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 41cb0ff6-35ee-49e8-9b31-962e733bda44
scatter(ranef(mm3)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Src to arousal (cor=$(get_trunc(mm3coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ 4100e32b-4b8d-4e02-8f3f-d994169a8c51
scatter( df2_subjlvl_hp.heights_phobia,ranef(mm3)[1][1,:], label="participant", title="Src to arousal (cor=$(get_trunc(mm3coef_i)))", ylabel="SCR cluster mean centering MLM raneff Intercept", xlabel="Heights Phobia")

# ╔═╡ c7d2bc63-47b4-45c6-aa5c-c1c890c4f502
scatter(ranef(mm3)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Src to arousal (cor=$(get_trunc(mm3coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ 128fefaa-9ec4-48c0-8df0-4afb0a92e7b7
md"## Model 2: arousal to fear"

# ╔═╡ b68b8257-8092-424e-a813-829b921dd881
mm4 = fit(LinearMixedModel, @formula(resp_fear ~ resp_arousal_cmc + resp_arousal_mean + (resp_arousal_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 8618f202-feb3-46c7-b329-1efa5c1bd39b
md"#### Arousal to fear, loglike: $(loglikelihood(mm4))"

# ╔═╡ e735c5a3-0fc1-44ce-9eab-791c4766f28e
md"#### Arousal to fear, AIC: $(aic(mm4))"

# ╔═╡ b89451ce-8cd9-46e4-9f4d-a883fd382b4f
mm4coef_i = cor(ranef(mm4)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 97497c87-9145-4655-980e-0cb9b6157397
mm4coef_w = cor(ranef(mm4)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 9218c7d0-5be6-4e8b-bdc2-5b4b1514c7d5
scatter(ranef(mm4)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Arousal to Fear (cor=$(get_trunc(mm4coef_i)))", xlabel="Arousal cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ 7946b928-1d29-43e9-8315-871b63bd49c5
scatter(ranef(mm4)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Arousal to Fear (cor=$(get_trunc(mm4coef_w)))", xlabel="Arousal cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ c7332e9c-e1df-4b78-b3d1-cb3ff2fcca71
md"## Model 3: scr to fear"

# ╔═╡ 330f8e14-843e-4d71-9b84-50eea6e4d534
mm6 = fit(LinearMixedModel, @formula(resp_fear ~ video_scr_cmc + video_scr_mean + (video_scr_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 04745b09-280b-4bef-87ce-69a2cd918ed6
md"#### Scr to fear, loglike: $(loglikelihood(mm6))"

# ╔═╡ 0a2cff38-c3b6-44d1-b63d-eccb7a199112
md"#### Scr to fear, AIC: $(aic(mm6))"

# ╔═╡ fd430c87-b690-4de3-a334-514dfdba0a66
mm6coef_i = cor(ranef(mm6)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 709a660d-df99-4c0e-b43c-b2ddd5ae3f87
mm6coef_w = cor(ranef(mm6)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ da14faf8-5d11-4b7d-a9f1-d82f4c1a73e2
scatter(ranef(mm6)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr to fear (cor=$(get_trunc(mm6coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ a4e8582c-9f88-4c61-8f86-120930344057
scatter(ranef(mm6)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr to fear (cor=$(get_trunc(mm6coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ 605709cf-56fb-4e7d-8da6-8944592a67a0
md"## Model 4: scr to valence"

# ╔═╡ 94b60f17-7950-41fa-9afc-f20a3e6c4447
mm12 = fit(LinearMixedModel, @formula(resp_valence ~ video_scr_cmc + video_scr_mean + (video_scr_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 2dee698f-ec02-400b-8656-0144acd093fb
md"#### Scr to arousal, loglike: $(loglikelihood(mm12))"

# ╔═╡ 93f4255e-fa75-42fd-87f7-bab88d393cc9
md"#### Scr to arousal, AIC: $(aic(mm12))"

# ╔═╡ 1592cb60-2fe1-488a-a721-a55e5a17bcc0
mm12coef_i = cor(ranef(mm12)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 4d61d49d-3247-4501-8022-c946124538da
mm12coef_w = cor(ranef(mm12)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 379eb0cf-1bed-4412-8a82-272ccf4ec172
scatter(ranef(mm12)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Src to valence (cor=$(get_trunc(mm12coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ a4ebcb4a-8f79-44b7-9984-c7213f606f0f
scatter(ranef(mm12)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Src to valence (cor=$(get_trunc(mm12coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ 974e7492-95a8-491e-b3e6-2626c9976fc6
md"## Model correlation comparison plots" 

# ╔═╡ 6e3111fd-7e6f-4702-9192-45f04767500b
plotdata3 = [mm3coef_i,mm4coef_i,mm6coef_i];

# ╔═╡ 43283800-f1c2-4550-9f62-df418b585217
plotcols3 = ["scr_to_aro_int","aro_to_fear_int", "scr_to_fear_int"];

# ╔═╡ b6e959d1-1da3-4fce-9d5f-71f5d5a855cd
plot(bar(plotcols3, plotdata3), title="Intercept term correlations to trait phobia", xlabel="Models", ylabel="Corr coef")

# ╔═╡ a7378ef0-9c1a-439c-8c19-61457e50428e
plotdata4 = [mm3coef_w,mm4coef_w,mm6coef_w];

# ╔═╡ f547bc31-acbc-4699-8ba3-0b6a546ca473
plotcols4 = ["scr_to_aro_beta","aro_to_fear_beta", "scr_to_fear_beta"];

# ╔═╡ ceec37f2-2b74-4a77-8741-abba5b598273
plot(bar(plotcols4, plotdata4), title="Beta/Weight term correlations to trait phobia", xlabel="Models", ylabel="Corr coef")

# ╔═╡ ee87be67-0b75-4b64-8cec-226f2cfac63c
models = [mm3,mm4,mm6];

# ╔═╡ 83a856ef-220e-4866-ad33-3878879f3498
plotcols5 = ["scr_to_aro","aro_to_fear", "scr_to_fear"];

# ╔═╡ 9b5114f8-26ce-44f0-aa17-b2087aad7551
plot(bar(plotcols5, models .|> loglikelihood), title="Log Likelihood model fit (higher = better)", xlabel="Models", ylabel="Log Likelihood")

# ╔═╡ 3e33f161-fc07-4531-9900-85d2a00d7ace
plot(bar(plotcols5, models .|> aic), title="AIC model fit (lower = better)", xlabel="Models", ylabel="AIC")

# ╔═╡ a25605fb-75bb-4bc6-8e20-8573c3113721
md"## Extra models"

# ╔═╡ db1a3e1f-647f-46c4-8d89-cad246be8293
md"#### Scr change to Arousal"

# ╔═╡ c6f84d25-31e3-433b-a9ef-9726463b5fd0
mm5 = fit(LinearMixedModel, @formula(resp_arousal ~ scr_change_video_cmc + scr_change_video_mean + (scr_change_video_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 382dd63c-261c-4f7a-af28-89c37f7c13fa
loglikelihood(mm5)

# ╔═╡ 97bb96cc-1a0e-4e40-bcbf-f03fe956c96a
aic(mm5)

# ╔═╡ 678f1fe3-d3b9-4c22-acf3-ece6453cd855
mm5coef_i = cor(ranef(mm5)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 50adebf3-9a30-4e0b-af51-0cd645c0e351
mm5coef_w = cor(ranef(mm5)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ 2704cbb8-55ca-45fc-9a43-f713c6a19650
scatter(ranef(mm5)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr change to arousal (cor=$(get_trunc(mm5coef_i)))", xlabel="SCR change cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ 47f40223-b93f-406a-9b38-494ff469c72c
scatter(ranef(mm5)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr change to arousal (cor=$(get_trunc(mm5coef_w)))", xlabel="SCR change cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ c503eb01-25cb-47a5-82bc-62b298db0195
md"#### Scr change to Fear"

# ╔═╡ b45bb383-1db8-4a4d-ac07-ca09fc52f138
mm7 = fit(LinearMixedModel, @formula(resp_fear ~ scr_change_video_cmc + scr_change_video_mean + (scr_change_video_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ c175dc18-e3be-4c7a-80c9-b07abcac8202
loglikelihood(mm7)

# ╔═╡ ccd6178b-bd27-4767-94e6-ec04918716b1
mm7coef_i = cor(ranef(mm7)[1][1,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ db89dd86-fa57-4f0a-acb2-5fc1a872c28f
mm7coef_w = cor(ranef(mm7)[1][2,:], df2_subjlvl_hp.heights_phobia);

# ╔═╡ baee8637-8a47-4438-bbcc-05c88f6117fa
scatter(ranef(mm7)[1][1,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr change to fear (cor=$(get_trunc(mm7coef_i)))", xlabel="SCR change cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ fe1dff56-5ab2-4b1b-9014-d4524ac536f4
scatter(ranef(mm7)[1][2,:], df2_subjlvl_hp.heights_phobia, label="participant", title="Scr change to fear (cor=$(get_trunc(mm7coef_w)))", xlabel="SCR change cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ a9e46b0a-3c7d-4c23-a903-d17062429086
md"#### Scr to Arousal with heights phobia modulating term"

# ╔═╡ 4c88655f-3208-4f24-9307-78028e877770
mm8 = fit(LinearMixedModel, @formula(resp_arousal ~ video_scr_cmc * heights_phobia + video_scr_mean + (video_scr_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ b25681e3-d89c-4091-a004-a74fa778b68f
loglikelihood(mm8)

# ╔═╡ 07b95a2c-c3c4-4300-818d-eb7c7509738e
aic(mm8)

# ╔═╡ 00702569-5052-4582-abea-abc0663408af
mm8coef_i = cor(ranef(mm8)[1][1,:], df2_subjlvl_hp.heights_phobia)

# ╔═╡ 509167a3-5aa8-4871-a838-16fd75225a6e
mm8coef_w = cor(ranef(mm8)[1][2,:], df2_subjlvl_hp.heights_phobia)

# ╔═╡ 008ea8e8-0500-4a58-8374-ff6670cd2cd4
mm9= fit(LinearMixedModel, @formula(resp_fear ~ resp_arousal_cmc * heights_phobia + resp_arousal_mean + (resp_arousal_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 92523847-a322-425f-9e41-f6799f4bfae9
loglikelihood(mm9)

# ╔═╡ 6c511ad7-a325-41fa-a63f-86292cbd58d0
mm10= fit(LinearMixedModel, @formula(resp_fear ~ video_scr_cmc * heights_phobia + video_scr_mean + (video_scr_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ fda8f196-e9b0-4e1b-a4d0-b61109a8769c
loglikelihood(mm10)

# ╔═╡ 5f5bccbb-fd01-4806-8dd0-b729f34e04d7
mm11= fit(LinearMixedModel, @formula(resp_fear ~ video_scr_cmc * heights_phobia * resp_arousal_cmc + video_scr_mean + resp_arousal_mean +  (video_scr_cmc * resp_arousal_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 46ab0a1a-0716-4400-be3d-85f401f04310
mm13= fit(LinearMixedModel, @formula(resp_fear ~ heights_phobia * video_scr_cmc * resp_arousal_cmc * resp_valence_cmc + video_scr_mean + resp_arousal_mean + resp_valence_mean + (video_scr_cmc * resp_arousal_cmc * resp_valence_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 88d2c0ec-6b3d-4388-98f8-248007a3555e
loglikelihood(mm11)

# ╔═╡ d13ee99a-d995-43cc-a32f-7d8c03af435d
loglikelihood(mm13)

# ╔═╡ 62e80efa-6097-49c2-9d29-fd9029948bb0
aic(mm11)

# ╔═╡ e8821396-a9ad-475b-b8e4-2f8167549632
aic(mm13)

# ╔═╡ 32a3c208-e406-43d5-bea7-ddb0b3b67061
mm14 = fit(LinearMixedModel, @formula(resp_fear ~ heights_phobia * video_scr_cmc * video_hp_cmc * resp_arousal_cmc * resp_valence_cmc + video_scr_mean + video_hp_mean + resp_arousal_mean + resp_valence_mean + (video_scr_cmc * video_hp_cmc * resp_arousal_cmc * resp_valence_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ dfe86456-47ac-47f2-83f0-a224e75b69c7
mm15 = fit(LinearMixedModel, @formula(resp_fear ~ heights_phobia * video_scr_cmc + heights_phobia *  video_hp_cmc + heights_phobia * resp_arousal_cmc + heights_phobia * resp_valence_cmc + video_scr_mean + video_hp_mean + resp_arousal_mean + resp_valence_mean + (video_scr_cmc + video_hp_cmc + resp_arousal_cmc + resp_valence_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ c236feb2-d67c-4052-87f6-87ce7e9cc7f8
mm16 = fit(LinearMixedModel, @formula(resp_fear ~ heights_phobia * video_scr_cmc + heights_phobia *  video_hp_cmc + heights_phobia * resp_arousal_cmc + heights_phobia * resp_valence_cmc + video_scr_mean + video_hp_mean + resp_arousal_mean + resp_valence_mean + (video_scr_cmc * video_hp_cmc * resp_arousal_cmc * resp_valence_cmc|participant_num_str)), dfcleancol2)

# ╔═╡ 3ceeb6a7-5556-4545-a6a4-3c3321807356
md"# Plot Test"

# ╔═╡ 5b13ea89-d498-432a-a203-996e0fb7f207
dfcleancol2

# ╔═╡ efc90c60-753e-4067-a8ad-46c4e568ee2f
dfcleancol2.resp_arousal

# ╔═╡ d165a61d-6f83-463e-b4ad-6afbd3fa2013
mm3

# ╔═╡ 6a9e15ae-531d-4953-85e1-aba8b4a0790e
randeff = only(ranef(mm3))

# ╔═╡ 7eb257f1-8b3b-4d14-a0dc-5e56cc221f76
randeff[1,:]

# ╔═╡ e31aeb3b-460a-4352-a579-4074cd926ef8
std(only(ranef(mm3)), dims=2)

# ╔═╡ df419173-257f-4143-a10b-31192c716449
scatter(dfcleancol2.video_scr, dfcleancol2.resp_arousal)

# ╔═╡ 599e3fd3-8810-4ffc-9af9-62f5b43385d5
maximum(dfcleancol2.video_scr)

# ╔═╡ c2ea34dd-f12f-41e2-a8da-9df58145a475
minimum(dfcleancol2.video_scr)

# ╔═╡ 460c3b4d-de87-45dd-add1-7516471351f6
xax = LinRange(minimum(dfcleancol2.video_scr),maximum(dfcleancol2.video_scr),10)

# ╔═╡ a9dc9660-ac83-446c-ba8d-80f0b6cd489b
inter = coef(mm3)[1]

# ╔═╡ 9812c713-5b5a-42ce-b93e-fadcfbcb86cf
cmcbeta = coef(mm3)[2]

# ╔═╡ ff504a2f-c14b-4132-9d0b-e06f61d69a08
meanbeta = coef(mm3)[3]

# ╔═╡ cb144cd3-ca7f-44dd-a921-b33cbf4b675e
yax = inter .+ ((cmcbeta .* xax) + (meanbeta .* xax))

# ╔═╡ 2e443fe1-01ee-4638-b0cd-e1d42e9e9314
df2_subjlvl_hp.heights_phobia

# ╔═╡ 2ec4d4c9-a20a-4194-8f3c-d66b0989fd12
std(df2_subjlvl_hp.heights_phobia)

# ╔═╡ f64bd09e-9652-40a4-894c-98fdb9559c63
std_up = mean(df2_subjlvl_hp.heights_phobia) + std(df2_subjlvl_hp.heights_phobia)

# ╔═╡ 3284ffe2-82c2-4422-a05c-e8993bf2dc1d
bm_std_up = df2_subjlvl_hp.heights_phobia .> std_up

# ╔═╡ ec3fb584-b839-4402-95e0-cb2f23140418
mean(randeff[1,bm_std_up])

# ╔═╡ 21c5b893-3f56-43ec-bd3c-24708c915656
std_dwn = mean(df2_subjlvl_hp.heights_phobia) - std(df2_subjlvl_hp.heights_phobia)

# ╔═╡ 7d1a511f-c0c1-4412-b644-04961560cc68
bm_std_dwn = df2_subjlvl_hp.heights_phobia .< std_dwn

# ╔═╡ 27f0c599-6b68-446b-911c-76b0d6011131
mean(randeff[1,bm_std_dwn])

# ╔═╡ e0040189-d6de-48f0-b7b0-03731701255a
yax_up = mean(randeff[1,bm_std_up]) .+ (mean(randeff[2,bm_std_up]).* xax)

# ╔═╡ 35f056c2-1137-4240-8c4c-cf322a47a34f
yax_dwn = abs.(mean(randeff[1,bm_std_dwn]) .+ (mean(randeff[2,bm_std_dwn]).* xax))

# ╔═╡ a94968e1-1c9a-4a88-8e37-b5aeec69d8f2
df = DataFrame(xax = xax, yax = yax, yax_up = yax_up, yax_dwn = yax_dwn)

# ╔═╡ 6ffa3226-3e5a-43a9-993b-c6f67bff54b4
scatter(dfcleancol2.video_scr, dfcleancol2.resp_arousal)

# ╔═╡ ea211a06-01cf-45d3-941d-64dc6162aaff
@df df plot!(:xax, :yax, ribbon=(:yax_dwn,:yax_up))

# ╔═╡ e6162568-4e9d-41ba-90b3-a42f45358a86
mm3

# ╔═╡ ad4a0e7d-1039-4e84-a0b0-ff2d208eccf1
ranef(mm3)

# ╔═╡ 146d62ef-89c9-4ba7-944b-044c236916c3


# ╔═╡ 5718cd05-16d1-44e5-84d3-cc7b67620266
md"# Plot Generic"

# ╔═╡ 4cf0a4ae-7d57-4037-a74f-dfe83b36cf8b
function plot_mm(mm, varx, vary)
	randeff = only(ranef(mm))
	xax = LinRange(minimum(varx),maximum(varx),10)
	inter = coef(mm)[1]
	cmcbeta = coef(mm)[2]
	meanbeta = coef(mm)[3]
	yax = inter .+ ((cmcbeta .* xax) + (meanbeta .* xax))
	std_up = mean(df2_subjlvl_hp.heights_phobia) + std(df2_subjlvl_hp.heights_phobia)
	bm_std_up = df2_subjlvl_hp.heights_phobia .> std_up
	std_dwn = mean(df2_subjlvl_hp.heights_phobia) - std(df2_subjlvl_hp.heights_phobia)
	bm_std_dwn = df2_subjlvl_hp.heights_phobia .< std_dwn
	yax_up = yax + (mean(randeff[1,bm_std_up]) .+ (mean(randeff[2,bm_std_up]).* xax))
	yax_dwn = yax + (mean(randeff[1,bm_std_dwn]) .+ (mean(randeff[2,bm_std_dwn]).* xax))
	df = DataFrame(xax = xax, yax = yax, yax_up = yax_up, yax_dwn = yax_dwn)
	scatter(varx, vary)
	@df df plot!(:xax, :yax)
	@df df plot!(:xax, :yax_dwn)
	@df df plot!(:xax, :yax_up)
end

# ╔═╡ b1ec26e2-4310-4c07-9768-f9681c00f047
plot_mm(mm3, dfcleancol2.video_scr, dfcleancol2.resp_arousal)

# ╔═╡ 332e4bc3-76a7-4b6f-b6e8-3ef80520d998
plot_mm(mm4, dfcleancol2.resp_arousal, dfcleancol2.resp_fear)

# ╔═╡ c420393d-9312-4373-8cd0-a27f7044b8a4
plot_mm(mm6, dfcleancol2.video_scr, dfcleancol2.resp_fear)

# ╔═╡ de788807-a952-49f5-a4f1-2b404fa516b2
md"## Mixed Models - With z-scores"

# ╔═╡ ebd88e51-6ba5-415f-9fb5-b39a5917ae32
df2cleancol2 = @pipe df2cleancol |> 
            groupby(_,:participant_num_str) |> # group by participant
			transform(_, [:resp_arousal_z, :video_scr_z] .=> mean)|>
			transform(_, [:resp_arousal_z, :resp_arousal_z_mean] => ByRow(-) => :resp_arousal_z_cmc, [:video_scr_z, :video_scr_z_mean]=> ByRow(-) => :video_scr_z_cmc) |> dropmissing

# ╔═╡ 64532f28-f71e-4148-accf-594007df031d
function dropnan(A)
	boolmask = (!).(any.(eachrow(isnan.(A))))
	return A[boolmask,:]
end

# ╔═╡ f851bb56-5db7-46e4-b2fa-18f2dcd8f6a7
function dropnanbm(A)
	boolmask = (!).(any.(eachrow(isnan.(A))))
	return boolmask
end

# ╔═╡ 10c51799-7b2b-4e81-8fdf-ccdeaad72080
bmask = dropnanbm(df2cleancol2[:,[:resp_arousal_z_mean, :resp_arousal_z_cmc,:video_scr_z_mean,:video_scr_z_cmc]])

# ╔═╡ fb81b450-505c-48cd-8d93-4e43b076c304
df2cleancol3 = df2cleancol2[bmask,:]

# ╔═╡ 770ff6be-a8fe-44fa-8fed-152475888357
gdf3 = groupby(df2cleancol3, :participant_num_str);

# ╔═╡ 4de7f1f4-2dde-4061-ab4f-8f9f1a1e71cc
hparr = combine(gdf3, :heights_phobia => mean, renamecols=false).heights_phobia;

# ╔═╡ d3243691-63a9-4085-9bb2-ab982c962077
md"## Zcored Model 1: src to arousal"

# ╔═╡ e555a176-42d8-4635-a9ea-6638ccb94eb8
mmz1 = fit(LinearMixedModel, @formula(resp_arousal_z ~ video_scr_z_cmc + video_scr_z_mean + (video_scr_z_cmc|participant_num_str)), df2cleancol3 )

# ╔═╡ 215d7f04-287b-46fc-9115-05bf35895cfd
md"#### Zscored Scr to arousal, loglike: $(loglikelihood(mmz1))"

# ╔═╡ 5b4c2b78-ff46-4e3b-9b26-75e9bf1db474
md"#### Zscored Scr to arousal, AIC: $(aic(mmz1))"

# ╔═╡ 853c22d0-ed90-4312-b10c-286ae390bebb
mmz1coef_w = cor(ranef(mmz1)[1][2,:], hparr)

# ╔═╡ ff8fa068-9c07-474b-874b-46015964eca0
mmz1coef_i = cor(ranef(mmz1)[1][1,:], hparr)

# ╔═╡ 28be242b-7636-4073-aa09-2895d2515d38
scatter(ranef(mmz1)[1][1,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz1coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ fd2ea288-a9d4-4905-879a-0608b69c806c
scatter(ranef(mmz1)[1][2,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz1coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ c39b1c68-46f6-4eee-8811-7b18a919190c
md"## Zscored Model 2: arousal to fear"

# ╔═╡ 08907dac-9072-4de8-b53c-c30f87d1e296
mmz2 = fit(LinearMixedModel, @formula(resp_fear_z ~ resp_arousal_z_cmc + resp_arousal_z_mean + (resp_arousal_z_cmc|participant_num_str)), df2cleancol3)

# ╔═╡ 72ab7b20-9114-4d63-a703-bbe092cbd9c7
md"#### Zscored Scr to arousal, loglike: $(loglikelihood(mmz2))"

# ╔═╡ 9e110999-9097-4021-a6a7-eabbfea157bb
md"#### Zscored Scr to arousal, AIC: $(aic(mmz2))"

# ╔═╡ dedd5e0a-45ee-4cb3-b23f-f339cf9ca79e
mmz2coef_w = cor(ranef(mmz2)[1][2,:], hparr)

# ╔═╡ 1cad4f45-7846-4f6f-af0f-332d8b6b8e45
mmz2coef_i = cor(ranef(mmz2)[1][1,:], hparr)

# ╔═╡ 634b4506-bc48-4ce5-9b49-92ae62b5a095
scatter(ranef(mmz2)[1][1,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz2coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ 52bd6b4a-7d0d-4a88-965b-d805b703c539
scatter(ranef(mmz2)[1][2,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz2coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ 8ac4591a-7e13-4b1c-92f1-c730f1ce3459
md"## Zscored Model 3: scr to fear"

# ╔═╡ f88d8ca8-39d7-4c7e-a68d-838137670617
mmz3 = fit(LinearMixedModel, @formula(resp_fear_z ~ video_scr_z_cmc + video_scr_z_mean + (video_scr_z_cmc|participant_num_str)), df2cleancol3)

# ╔═╡ 2f936439-5413-48c7-bdad-670c3328f057
md"#### Scr to fear, loglike: $(loglikelihood(mmz3))"

# ╔═╡ 11490f5f-26a7-4cfd-b04f-88ae30c96268
md"#### Scr to fear, AIC: $(aic(mmz3))"

# ╔═╡ 7aed13c1-068e-4bf8-ad05-7bf780c18ca4
mmz3coef_w = cor(ranef(mmz3)[1][2,:], hparr)

# ╔═╡ 096da4d1-7df3-457b-b818-e605adb56cd9
mmz3coef_i = cor(ranef(mmz3)[1][1,:], hparr)

# ╔═╡ 088e4121-9138-4dbe-817b-f871223be945
scatter(ranef(mmz3)[1][1,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz3coef_i)))", xlabel="SCR cluster mean centering MLM raneff Intercept", ylabel="Heights Phobia")

# ╔═╡ 729fe2cc-42ae-449c-8626-37b486fd77fe
scatter(ranef(mmz3)[1][2,:], hparr, label="participant", title="Src to arousal (cor=$(get_trunc(mmz3coef_w)))", xlabel="SCR cluster mean centering MLM raneff Weight", ylabel="Heights Phobia")

# ╔═╡ a91e0afd-c986-4cae-9803-1879265476ff
md"## Zscored Model correlation comparison plots" 

# ╔═╡ eb11d2e7-1b23-4bf7-8007-5bb6a05377d3
plotdata_z = [mmz1coef_i,mmz2coef_i,mmz3coef_i];

# ╔═╡ 60b2c197-9cad-4816-bc79-0a14fc6f792b
plotcols_z = ["scr_to_aro_int","aro_to_fear_int", "scr_to_fear_int"];

# ╔═╡ 299c67a7-d0d4-4e64-bedf-86ab95576705
plot(bar(plotcols_z, plotdata_z), title="Intercept term correlations to trait phobia", xlabel="Models", ylabel="Corr coef")

# ╔═╡ b970a3a2-5b32-41a4-a319-e1e2d9ce0061
plotdata_z1 = [mmz1coef_w,mmz2coef_w,mmz3coef_w];

# ╔═╡ 1b1bcc0a-5709-48a5-9135-a54b6ea35338
plotcols_z1 = ["scr_to_aro_beta","aro_to_fear_beta", "scr_to_fear_beta"];

# ╔═╡ 2eb84e2a-dacb-4237-939f-0af669e3893e
plot(bar(plotcols_z1, plotdata_z1), title="Beta/Weight term correlations to trait phobia", xlabel="Models", ylabel="Corr coef")

# ╔═╡ 641d9de2-357f-411a-92b6-e5caed1fc812
models2 = [mmz1,mmz2,mmz3];

# ╔═╡ 162ffde5-73a6-494a-8ebe-0683eca86fec
plotcols_z3 = ["scr_to_aro","aro_to_fear", "scr_to_fear"];

# ╔═╡ e912cf0e-3fdc-47ea-adbc-0381e320d97a
plot(bar(plotcols_z3, models2 .|> loglikelihood), title="Log Likelihood model fit (higher = better)", xlabel="Models", ylabel="Log Likelihood")

# ╔═╡ d54adcc0-8202-421c-be77-16edffd4f5a8
plot(bar(plotcols_z3, models2 .|> aic), title="AIC model fit (lower = better)", xlabel="Models", ylabel="AIC")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
NaNStatistics = "b946abbf-3ea7-4610-9019-9858bfdeaf2d"
Pipe = "b98c9c47-44ae-5843-9183-064241ee97a0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[compat]
CSV = "~0.10.11"
DataFrames = "~1.6.1"
GLM = "~1.9.0"
MixedModels = "~4.22.1"
NaNStatistics = "~0.6.31"
Pipe = "~1.3.0"
Plots = "~1.39.0"
StatsPlots = "~0.15.6"
Tables = "~1.11.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "19159d6d141bede97901d69eb579a0c12d94f83a"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9a731850434825d183af39c6e6cd0a1c32dd7e20"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.4.2"

[[deps.Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "ConcurrentUtilities", "DataAPI", "Dates", "EnumX", "LoggingExtras", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "954666e252835c4cf8819ce4ffaf31073c1b7233"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "2.6.2"

[[deps.ArrowTypes]]
deps = ["Sockets", "UUIDs"]
git-tree-sha1 = "8c37bfdf1b689c6677bbfc8986968fe641f6a299"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "2.2.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BSplineKit]]
deps = ["ArrayLayouts", "BandedMatrices", "FastGaussQuadrature", "LinearAlgebra", "PrecompileTools", "Random", "Reexport", "SparseArrays", "Static", "StaticArrays", "StaticArraysCore"]
git-tree-sha1 = "e8c349b71f1cde3faad14bb09d3fc5a3b287eeb8"
uuid = "093aae92-e908-43d7-9660-e50ee39d5a0a"
version = "0.16.5"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "8f32ba3789b29880901748dce28f7d5c1d4ae86a"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "a55462dfddabc34bc97d3a7403a2ca2802179ae6"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.3.1"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"

[[deps.ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "05f9816a77231b07e634ab8715ba50e5249d6f76"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.5"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "c0ae2a86b162fb5d7acc65269b469ff5b8a73594"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.1"

[[deps.CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "8bf4f9e2ee52b5e217451a7cd9171fcd4e16ae23"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.CodecZstd]]
deps = ["CEnum", "TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "849470b337d0fa8449c21061de922386f32949d9"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5225c965635d8c21168e32a12954675e7bea1151"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.10"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "93ff6a4d5e7bfe27732259bfabbdd19940d8af1f"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "95220473901735a0f4df9d1ca5b171b568b2daa3"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.13.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "SpecialFunctions", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6c26c5e8a4203d43b5497be3ec5d4e0c3cde240a"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.4+0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "5c9f1e635e8d491297e596b56fec1c95eafb95a3"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.20.1"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.MixedModels]]
deps = ["Arrow", "BSplineKit", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "PrecompileTools", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables", "TypedTables"]
git-tree-sha1 = "2dc021878892ed15b0bcd394d9b158e40c60217f"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.22.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "4cc0c5a83933648b615c36c2b956d94fda70641e"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.7"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "6985021d02ab8c509c841bb8b2becd3145a7b490"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.3"

[[deps.NLopt]]
deps = ["MathOptInterface", "NLopt_jll"]
git-tree-sha1 = "19d2a1c8a3c5b5a459f54a10e54de630c4a05701"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.0.0"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NaNStatistics]]
deps = ["IfElse", "LoopVectorization", "PrecompileTools", "Static"]
git-tree-sha1 = "a0e3ceee48f18b00ff5e34ec51646fb5c0cccf61"
uuid = "b946abbf-3ea7-4610-9019-9858bfdeaf2d"
version = "0.6.31"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "66b2fcd977db5329aa35cac121e5b94dd6472198"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.28"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6842ce83a836fbbc0cfeca0b5a4de1a4dcbdb8d1"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.8"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "48f393b0231516850e39f6c756970e7ca8b77045"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.2"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "5cf6c4583533ee38639f73b880f35fc85f2941e0"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.3"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TZJData]]
deps = ["Artifacts"]
git-tree-sha1 = "d39314cdbaf5b90a047db33858626f8d1cc973e1"
uuid = "dc5dba14-91b3-4cab-a142-028a31da12f7"
version = "1.0.0+2023c"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimeZones]]
deps = ["Artifacts", "Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Scratch", "TZJData", "Unicode", "p7zip_jll"]
git-tree-sha1 = "89e64d61ef3cd9e80f7fc12b7d13db2d75a23c03"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.13.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.TypedTables]]
deps = ["Adapt", "Dictionaries", "Indexing", "SplitApplyCombine", "Tables", "Unicode"]
git-tree-sha1 = "d911ae4e642cf7d56b1165d29ef0a96ba3444ca9"
uuid = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "f7c281e9c61905521993a987d38b5ab1d4b53bef"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═27ba71bf-1f2e-4a79-89c7-2ccdbf3ec6f8
# ╠═ffdbec30-2bd7-11ee-2284-1bb2a6d11973
# ╠═bf0aca2a-99d2-4ef0-a1e4-aa4e48e265fc
# ╠═8d099526-be2e-4543-bece-73721bdce561
# ╠═013bfd5b-8c78-4f6b-8d6b-fd83ace08482
# ╠═f16dac04-f3a5-48f6-8eb1-46dd0abe16d4
# ╠═2d64c863-ed4f-466d-bcf7-882397d7bf5e
# ╠═8b8de4be-9f00-4a86-8151-1746d307f40f
# ╠═313f2cd7-9dd6-462c-83e6-86232e8dfa9e
# ╠═2d0f1009-7f92-4f91-bc26-2a7cd5b4ec5e
# ╟─277630ab-e88d-43bf-a511-a33ab23cddb1
# ╟─9a2edcaa-93d4-44d0-909b-b4f6e3f548f7
# ╠═212a249e-6b19-4ddf-9125-93edd3b36977
# ╠═c214f257-a167-4dd5-bc20-41386c398001
# ╠═0201632a-e3e0-447d-a6a8-5965e1570a64
# ╠═5e5da2db-5aac-4296-abad-4535ff669e8c
# ╠═e9eab7cf-e11f-45a3-a85b-4fde496cfb2d
# ╠═126b8f45-ae49-4bc9-bf6e-c2425d864ce2
# ╠═aec29b59-5cf5-4b2d-9853-12f5e7350f2c
# ╠═d4079ef4-5441-4072-a055-fc372b89843f
# ╠═229d3ba7-9a67-489f-9043-ccf1134d9de1
# ╠═e00e776d-3f7d-43d8-8da8-d58419234802
# ╠═21b5dbbb-cd33-475a-96a7-6a0d2ff0e0d5
# ╠═7964d5b3-c711-4feb-948f-f55853b4e33a
# ╠═c6bd9ec5-dc72-4e3e-a843-6ef485bb614c
# ╠═515c6bc3-a59b-4b71-acae-e639e1a9bd2a
# ╠═0f6bf3f7-bbe5-47ae-a765-82f8101c44ff
# ╟─53807a24-0fd1-44c3-9808-4f74fa2d3fe9
# ╠═63522390-0eea-4ccc-9215-5a54c1642511
# ╠═8f813e05-4dc3-48e4-af79-bafe2b643338
# ╠═d5a188fb-8fc4-458d-a998-c89593d87c38
# ╠═d940c157-8556-4d4a-92cc-380b3a7942b7
# ╠═323785ee-1332-47a5-8726-f25aa002ff81
# ╠═409c55aa-a290-4403-a707-42e6b39c1943
# ╠═6be6b7ec-f21d-4fa2-b99b-761bdc55c8ee
# ╠═b0263688-0f15-47b7-bf66-299917962941
# ╠═70c5c4cc-087a-4f24-a2ff-3b1349e969b1
# ╠═9c98b6f4-9c04-44e3-ad4a-539ce588f9f4
# ╠═dc357f8d-1960-476e-a728-4975f9ec4444
# ╠═c80d8246-19c2-4c63-8301-861e52c6f6ed
# ╠═3bfd7f99-f2a5-4d02-a373-7d022ed1f877
# ╠═9be1cff9-3cfd-4299-976b-8adeeeb5b54f
# ╠═04cad377-0e86-4baa-bf7c-0a738fa973df
# ╠═5dd5cbf6-df2e-414a-8e24-0d767eaf1af7
# ╠═b18892c8-3ba5-4558-af64-9c915dff878a
# ╠═e1489517-53a0-4b33-bcfe-480173139711
# ╠═d6016ec7-8018-4916-86bc-64d1598f8d06
# ╠═21b79764-fa90-4fe9-924f-80c22919a90f
# ╠═e2cc8c45-dc01-44e6-b2d9-c180c5a9cd58
# ╠═39af73ef-b8de-4af5-9b67-f3f9581384ed
# ╟─d699231e-48b1-40f7-abdb-a1f5c7954c0d
# ╠═d5dd7ff9-18b2-44ed-ade2-cc9dac386d44
# ╠═4d81be85-1611-41d6-907b-dcf411143162
# ╠═412538a0-2bb3-4e5d-a78c-65dfdcbbaa08
# ╠═19195589-73d3-48ae-9553-0c5ebaf0effe
# ╠═2df570fc-0f9e-4166-82d8-83fcdb115d2f
# ╠═4815493b-e5d4-4572-aa07-380da5a2c36d
# ╟─39534e93-a4b4-4c28-a7a2-deede208775a
# ╠═eae03171-3511-4ac5-8065-0b5090159e80
# ╠═d0aefb0a-7ae2-43f7-a0ea-fdf5662c2ca1
# ╠═39f10fb6-0383-4742-8ebf-fd0969aeda3c
# ╠═0c5fef51-9132-4a37-bfc3-ff042a82acc5
# ╠═1fa45402-b95c-4cbb-9eb3-dc1e3ec169a8
# ╠═7dcb34da-5d5e-4389-9757-e5ea1ff38263
# ╠═a4186efc-8752-4bbb-86ed-0160a1ed9359
# ╠═299a56a3-5b2f-486e-afdc-2e8adac87962
# ╟─cc7e0d56-68a6-44e0-b227-784d36abc89f
# ╠═1d3ce898-73a8-4b17-a758-bceaf5d5bb05
# ╠═29bd56bb-757b-40d1-ad46-9874418974d5
# ╟─3b9e3928-fef5-446a-b0dc-ba2371a2720a
# ╟─85655803-cef4-4a6f-a9c6-3daf41e93ecb
# ╠═46020a9f-a305-43f5-87e3-c5f827da27b5
# ╠═1cf2c58b-22db-4271-bb4e-cd93f99058b1
# ╠═b1362f6d-d74c-4ee6-b5a3-7097e9600fc5
# ╠═1dc2d037-0b5a-46d9-97bd-cc220d022411
# ╠═9cd7c5e9-b8f2-4cd8-a3b5-5dc632231a64
# ╠═aa21b5ee-e94f-4063-b133-3dccee4bab2c
# ╟─3789cddf-a693-4114-880a-76c531947086
# ╠═51cb82d5-6ad3-455d-972b-4b914a3dcdc7
# ╠═7bd4ace3-c5f8-4cd4-bd92-d5ec73da2d40
# ╟─dc95e5cf-3eb2-4127-8730-e3444568639a
# ╠═69a13bd2-4154-4897-a964-b7eee3998b24
# ╠═97b01db8-8d58-4ab9-9f20-634507db6f97
# ╠═13b0dee5-7fb2-4196-9342-4771edf287ac
# ╠═1da4269e-ee2d-49b8-b537-248f912940cf
# ╠═0822037d-4284-4e00-8a35-9c2de8f6f4e3
# ╠═41cb0ff6-35ee-49e8-9b31-962e733bda44
# ╠═4100e32b-4b8d-4e02-8f3f-d994169a8c51
# ╠═c7d2bc63-47b4-45c6-aa5c-c1c890c4f502
# ╠═128fefaa-9ec4-48c0-8df0-4afb0a92e7b7
# ╠═b68b8257-8092-424e-a813-829b921dd881
# ╟─8618f202-feb3-46c7-b329-1efa5c1bd39b
# ╟─e735c5a3-0fc1-44ce-9eab-791c4766f28e
# ╟─b89451ce-8cd9-46e4-9f4d-a883fd382b4f
# ╟─97497c87-9145-4655-980e-0cb9b6157397
# ╟─9218c7d0-5be6-4e8b-bdc2-5b4b1514c7d5
# ╠═7946b928-1d29-43e9-8315-871b63bd49c5
# ╠═c7332e9c-e1df-4b78-b3d1-cb3ff2fcca71
# ╠═330f8e14-843e-4d71-9b84-50eea6e4d534
# ╠═04745b09-280b-4bef-87ce-69a2cd918ed6
# ╠═0a2cff38-c3b6-44d1-b63d-eccb7a199112
# ╠═fd430c87-b690-4de3-a334-514dfdba0a66
# ╠═709a660d-df99-4c0e-b43c-b2ddd5ae3f87
# ╠═da14faf8-5d11-4b7d-a9f1-d82f4c1a73e2
# ╠═a4e8582c-9f88-4c61-8f86-120930344057
# ╠═605709cf-56fb-4e7d-8da6-8944592a67a0
# ╠═94b60f17-7950-41fa-9afc-f20a3e6c4447
# ╠═2dee698f-ec02-400b-8656-0144acd093fb
# ╠═93f4255e-fa75-42fd-87f7-bab88d393cc9
# ╠═1592cb60-2fe1-488a-a721-a55e5a17bcc0
# ╠═4d61d49d-3247-4501-8022-c946124538da
# ╠═379eb0cf-1bed-4412-8a82-272ccf4ec172
# ╠═a4ebcb4a-8f79-44b7-9984-c7213f606f0f
# ╠═974e7492-95a8-491e-b3e6-2626c9976fc6
# ╠═6e3111fd-7e6f-4702-9192-45f04767500b
# ╠═43283800-f1c2-4550-9f62-df418b585217
# ╠═b6e959d1-1da3-4fce-9d5f-71f5d5a855cd
# ╠═a7378ef0-9c1a-439c-8c19-61457e50428e
# ╠═f547bc31-acbc-4699-8ba3-0b6a546ca473
# ╠═ceec37f2-2b74-4a77-8741-abba5b598273
# ╠═ee87be67-0b75-4b64-8cec-226f2cfac63c
# ╠═83a856ef-220e-4866-ad33-3878879f3498
# ╠═9b5114f8-26ce-44f0-aa17-b2087aad7551
# ╠═3e33f161-fc07-4531-9900-85d2a00d7ace
# ╟─a25605fb-75bb-4bc6-8e20-8573c3113721
# ╟─db1a3e1f-647f-46c4-8d89-cad246be8293
# ╠═c6f84d25-31e3-433b-a9ef-9726463b5fd0
# ╠═382dd63c-261c-4f7a-af28-89c37f7c13fa
# ╟─97bb96cc-1a0e-4e40-bcbf-f03fe956c96a
# ╟─678f1fe3-d3b9-4c22-acf3-ece6453cd855
# ╟─50adebf3-9a30-4e0b-af51-0cd645c0e351
# ╟─2704cbb8-55ca-45fc-9a43-f713c6a19650
# ╟─47f40223-b93f-406a-9b38-494ff469c72c
# ╟─c503eb01-25cb-47a5-82bc-62b298db0195
# ╠═b45bb383-1db8-4a4d-ac07-ca09fc52f138
# ╟─c175dc18-e3be-4c7a-80c9-b07abcac8202
# ╠═ccd6178b-bd27-4767-94e6-ec04918716b1
# ╟─db89dd86-fa57-4f0a-acb2-5fc1a872c28f
# ╠═baee8637-8a47-4438-bbcc-05c88f6117fa
# ╟─fe1dff56-5ab2-4b1b-9014-d4524ac536f4
# ╟─a9e46b0a-3c7d-4c23-a903-d17062429086
# ╠═4c88655f-3208-4f24-9307-78028e877770
# ╠═b25681e3-d89c-4091-a004-a74fa778b68f
# ╠═07b95a2c-c3c4-4300-818d-eb7c7509738e
# ╠═00702569-5052-4582-abea-abc0663408af
# ╠═509167a3-5aa8-4871-a838-16fd75225a6e
# ╠═008ea8e8-0500-4a58-8374-ff6670cd2cd4
# ╠═92523847-a322-425f-9e41-f6799f4bfae9
# ╠═6c511ad7-a325-41fa-a63f-86292cbd58d0
# ╠═fda8f196-e9b0-4e1b-a4d0-b61109a8769c
# ╠═5f5bccbb-fd01-4806-8dd0-b729f34e04d7
# ╠═46ab0a1a-0716-4400-be3d-85f401f04310
# ╠═88d2c0ec-6b3d-4388-98f8-248007a3555e
# ╠═d13ee99a-d995-43cc-a32f-7d8c03af435d
# ╠═62e80efa-6097-49c2-9d29-fd9029948bb0
# ╠═e8821396-a9ad-475b-b8e4-2f8167549632
# ╠═32a3c208-e406-43d5-bea7-ddb0b3b67061
# ╠═dfe86456-47ac-47f2-83f0-a224e75b69c7
# ╠═c236feb2-d67c-4052-87f6-87ce7e9cc7f8
# ╟─3ceeb6a7-5556-4545-a6a4-3c3321807356
# ╠═5b13ea89-d498-432a-a203-996e0fb7f207
# ╠═efc90c60-753e-4067-a8ad-46c4e568ee2f
# ╠═d165a61d-6f83-463e-b4ad-6afbd3fa2013
# ╠═6a9e15ae-531d-4953-85e1-aba8b4a0790e
# ╠═7eb257f1-8b3b-4d14-a0dc-5e56cc221f76
# ╠═e31aeb3b-460a-4352-a579-4074cd926ef8
# ╠═df419173-257f-4143-a10b-31192c716449
# ╠═599e3fd3-8810-4ffc-9af9-62f5b43385d5
# ╠═c2ea34dd-f12f-41e2-a8da-9df58145a475
# ╠═460c3b4d-de87-45dd-add1-7516471351f6
# ╠═a9dc9660-ac83-446c-ba8d-80f0b6cd489b
# ╠═9812c713-5b5a-42ce-b93e-fadcfbcb86cf
# ╠═ff504a2f-c14b-4132-9d0b-e06f61d69a08
# ╠═cb144cd3-ca7f-44dd-a921-b33cbf4b675e
# ╠═2e443fe1-01ee-4638-b0cd-e1d42e9e9314
# ╠═2ec4d4c9-a20a-4194-8f3c-d66b0989fd12
# ╠═f64bd09e-9652-40a4-894c-98fdb9559c63
# ╠═3284ffe2-82c2-4422-a05c-e8993bf2dc1d
# ╠═ec3fb584-b839-4402-95e0-cb2f23140418
# ╠═21c5b893-3f56-43ec-bd3c-24708c915656
# ╠═7d1a511f-c0c1-4412-b644-04961560cc68
# ╠═27f0c599-6b68-446b-911c-76b0d6011131
# ╠═e0040189-d6de-48f0-b7b0-03731701255a
# ╠═35f056c2-1137-4240-8c4c-cf322a47a34f
# ╠═a94968e1-1c9a-4a88-8e37-b5aeec69d8f2
# ╠═6ffa3226-3e5a-43a9-993b-c6f67bff54b4
# ╠═ea211a06-01cf-45d3-941d-64dc6162aaff
# ╠═e6162568-4e9d-41ba-90b3-a42f45358a86
# ╠═ad4a0e7d-1039-4e84-a0b0-ff2d208eccf1
# ╠═146d62ef-89c9-4ba7-944b-044c236916c3
# ╟─5718cd05-16d1-44e5-84d3-cc7b67620266
# ╠═4cf0a4ae-7d57-4037-a74f-dfe83b36cf8b
# ╠═b1ec26e2-4310-4c07-9768-f9681c00f047
# ╠═332e4bc3-76a7-4b6f-b6e8-3ef80520d998
# ╠═c420393d-9312-4373-8cd0-a27f7044b8a4
# ╟─de788807-a952-49f5-a4f1-2b404fa516b2
# ╠═ebd88e51-6ba5-415f-9fb5-b39a5917ae32
# ╠═64532f28-f71e-4148-accf-594007df031d
# ╠═f851bb56-5db7-46e4-b2fa-18f2dcd8f6a7
# ╠═10c51799-7b2b-4e81-8fdf-ccdeaad72080
# ╠═fb81b450-505c-48cd-8d93-4e43b076c304
# ╠═770ff6be-a8fe-44fa-8fed-152475888357
# ╠═4de7f1f4-2dde-4061-ab4f-8f9f1a1e71cc
# ╟─d3243691-63a9-4085-9bb2-ab982c962077
# ╠═e555a176-42d8-4635-a9ea-6638ccb94eb8
# ╟─215d7f04-287b-46fc-9115-05bf35895cfd
# ╟─5b4c2b78-ff46-4e3b-9b26-75e9bf1db474
# ╠═853c22d0-ed90-4312-b10c-286ae390bebb
# ╠═ff8fa068-9c07-474b-874b-46015964eca0
# ╠═28be242b-7636-4073-aa09-2895d2515d38
# ╠═fd2ea288-a9d4-4905-879a-0608b69c806c
# ╟─c39b1c68-46f6-4eee-8811-7b18a919190c
# ╠═08907dac-9072-4de8-b53c-c30f87d1e296
# ╠═72ab7b20-9114-4d63-a703-bbe092cbd9c7
# ╠═9e110999-9097-4021-a6a7-eabbfea157bb
# ╠═dedd5e0a-45ee-4cb3-b23f-f339cf9ca79e
# ╠═1cad4f45-7846-4f6f-af0f-332d8b6b8e45
# ╠═634b4506-bc48-4ce5-9b49-92ae62b5a095
# ╠═52bd6b4a-7d0d-4a88-965b-d805b703c539
# ╠═8ac4591a-7e13-4b1c-92f1-c730f1ce3459
# ╠═f88d8ca8-39d7-4c7e-a68d-838137670617
# ╠═2f936439-5413-48c7-bdad-670c3328f057
# ╠═11490f5f-26a7-4cfd-b04f-88ae30c96268
# ╠═7aed13c1-068e-4bf8-ad05-7bf780c18ca4
# ╠═096da4d1-7df3-457b-b818-e605adb56cd9
# ╠═088e4121-9138-4dbe-817b-f871223be945
# ╠═729fe2cc-42ae-449c-8626-37b486fd77fe
# ╠═a91e0afd-c986-4cae-9803-1879265476ff
# ╠═eb11d2e7-1b23-4bf7-8007-5bb6a05377d3
# ╠═60b2c197-9cad-4816-bc79-0a14fc6f792b
# ╠═299c67a7-d0d4-4e64-bedf-86ab95576705
# ╠═b970a3a2-5b32-41a4-a319-e1e2d9ce0061
# ╠═1b1bcc0a-5709-48a5-9135-a54b6ea35338
# ╠═2eb84e2a-dacb-4237-939f-0af669e3893e
# ╠═641d9de2-357f-411a-92b6-e5caed1fc812
# ╠═162ffde5-73a6-494a-8ebe-0683eca86fec
# ╠═e912cf0e-3fdc-47ea-adbc-0381e320d97a
# ╠═d54adcc0-8202-421c-be77-16edffd4f5a8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
