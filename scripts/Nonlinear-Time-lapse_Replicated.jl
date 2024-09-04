using DrWatson
@quickactivate "Nonlinear-JRM"

# Unix gcc config
ENV["DEVITO_LOGGING"]="ERROR" # Silence Devito verbosity
# ENV["DEVITO_ARCH"]  = "gcc" #  Use the gcc compiler
# ENV["DEVITO_LANGUAGE"] = "openmp"  # Use openmp for thread parallelism

# OSX clang config

# ENV["DEVITO_LOGGING"]="INFO" # Silence Devito verbosity
# ENV["DEVITO_ARCH"]  = clang #  Use the gcc compiler
# ENV["DEVITO_LANGUAGE"] = "openmp"  # Use openmp for thread parallelism
# ENV["DEVITO_PLATFORM"] = "m1"  # Specify we are on an m1 architecture to the Deivto compiler

# Set number of threads
ENV["OMP_NUM_THREADS"] = div(Sys.CPU_THREADS, 4)

using SlimPlotting, JUDI, JLD2, JOLI, SlimOptim, SegyIO, SetIntersectionProjection
using PyPlot, Images, Random, LinearAlgebra, StatsBase, DSP
using ArgParse
#matplotlib.use("agg")
Random.seed!(2023)

include(srcdir("Seis4CCSops.jl"))
include(srcdir("utils.jl"))

parsed_args = parse_commandline()
nv = parsed_args["nv"]
nv_id = parsed_args["nv_id"]
nsrc = parsed_args["nsrc"]
nrec = parsed_args["nrec"]
snr = parsed_args["snr"]
batchsize = parsed_args["batchsize"]
niter = parsed_args["niter"]
dt = parsed_args["dt"]
nt = parsed_args["nt"]
irate = parsed_args["irate"]
timeD = parsed_args["timeD"]
dtD = parsed_args["dtD"]
f0 = parsed_args["f0"]
inv_type = parsed_args["inv_type"]
replicate = parsed_args["replicate"]

data_path = "sims_jutul"
data_dict = @strdict dt nt irate 
@load datadir(joinpath(data_path, "velrho_10m_nJRM_"*savename(data_dict, "jld2"; digits=6))) v_stack rho_stack

v1, v2 = v_stack[1][1:4:end, 1:2:end], v_stack[nv_id][1:4:end, 1:2:end];
#rho1, rho2 = rho_stack[1][1:2:end, 1:2:end], rho_stack[(nv_id-1)*5][1:2:end, 1:2:end]
# Model size
n = size(v1);
d = (15.0f0, 15.0f0);
o = (0f0, 0f0);

idx_wb = maximum(find_water_bottom(v1.-v1[1,1]));

# Use this for impedance inversion
# v0 = deepcopy(v1)
# v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(5f0))
# m0 = (1f0./v0).^2f0
# m1 = (1f0./v1).^2f0
# m2 = (1f0./v2).^2f0
# rho0 = deepcopy(rho1)
# rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(5f0))
# imp0 = v0.*rho0
# dimp = [vec(imp0-v1.*rho1), vec(imp0-v2.*rho2)]

m_baseline = (1f0./v1).^2f0;
m_monitor = (1f0./v2).^2f0;
m0 = 1 .* m_baseline;
m0[:, idx_wb:end] = imfilter(m0[:, idx_wb:end].^(.5), Kernel.gaussian(5f0)).^2;
m0 = convert(Matrix{Float32}, m0);

# Get modeled data for inversion
space_order = 16;
data_dict = @strdict nsrc nrec nv nv_id snr timeD dtD f0 space_order replicate 
sim_name = "sims_seis";
if isfile(datadir(joinpath(sim_name,"dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6))))
    @load datadir(joinpath(sim_name,"dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6))) d_baseline qb
else
    d_baseline = F_baseline * qb  # True data for the baseline
    # Adding noise to data
    noiseb = deepcopy(d_baseline)
    for l = 1:nsrc
        noiseb.data[l] = randn(Float32, size(d_baseline.data[l]))
        noiseb.data[l] = real.(ifft(fft(noiseb.data[l]).*fft(qb.data[1])))
    end
    noiseb = noiseb/norm(noiseb) * norm(d_baseline) * 10f0^(-snr/20f0)
    d_baseline = d_baseline + noiseb
    data_dict = @strdict nsrc nrec nv nv_id snr timeD dtD f0 space_order replicate d_baseline qb
    @tagsave(
        datadir(joinpath(sim_name, "dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6))),
        data_dict;
        safe=true)
end

if isfile(datadir(joinpath(sim_name,"dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6))))
    @load datadir(joinpath(sim_name,"dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6))) d_monitor qm
else
    d_monitor = F_monitor * qm  # True data for the baseline
    # Adding noise to data
    noisem = deepcopy(d_monitor)
    for l = 1:nsrc
        noisem.data[l] = randn(Float32, size(d_monitor.data[l]))
        noisem.data[l] = real.(ifft(fft(noisem.data[l]).*fft(qm.data[1])))
    end
    noisem = noisem/norm(noisem) * norm(d_monitor) * 10f0^(-snr/20f0)
    d_monitor = d_monitor + noisem
    data_dict = @strdict nsrc nrec nv nv_id snr timeD dtD f0 space_order replicate d_monitor qm
    @tagsave(
        datadir(joinpath(sim_name, "dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6))),
        data_dict;
        safe=true)
end

# Acquisition parameters
timeD = timeD;  # We record the pressure at the receivers for 3000ms (3 Sec) for each source
dtD = dtD;   # We sample the measurements every 2 ms
f0 = f0;  # We use a 14.5Hz source (0.0145 KHz since we measure time in ms) and band pass it to a realitic frequency band
wavelet = filter_data(ricker_wavelet(timeD, dtD, f0), 4f0; fmin=3, fmax=15);

# Source coordinates in physical units (m)
xsrc1 = [vcat(qb[i].geometry.xloc...) for i = 1:d_baseline.nsrc];
xsrc2 = [vcat(qm[i].geometry.xloc...) for i = 1:d_monitor.nsrc];
xsrc = [xsrc1, xsrc2];
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc)); # Set y coordinate to zero since it's 2D
zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc));  # Sources at 15m depth

# JUDI geometry object
#srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
srcGeometry_b = Geometry(xsrc[1], ysrc, zsrc; dt=dtD, t=timeD);
srcGeometry_m = Geometry(xsrc[2], ysrc, zsrc; dt=dtD, t=timeD);

# Receivers (measurments) coordinates in physical units
xrec = d_baseline.geometry.xloc[1]; # range(start=120f0, stop=(n[1]-1)*d[1]-120f0,length=nrec)  # Receivers every 120m
yrec = d_baseline.geometry.yloc[1]; # 0f0 # WE have to set the y coordinate to zero (or any number) for 2D modeling
zrec = d_baseline.geometry.zloc[1]; # range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc);

model_baseline = Model(n, d, o, m_baseline; nb=80);
model_monitor = Model(n, d, o, m_monitor; nb=80);

space_order = 8;
opt = Options(dt_comp=1f0, space_order=space_order); # We fix the computational time-step to avoid slight variations in numerical scheme leading to differences in the data.
F_baseline = judiModeling(model_baseline, srcGeometry_b, recGeometry; options=opt);
F_monitor = judiModeling(model_monitor, srcGeometry_m, recGeometry; options=opt);

# Define the operator for the background
F0_baseline = F_baseline(;m=m0);  # F(m0 + dm1) with intial dm1=0
F0_monitor = F_monitor(;m=m0); # F(m0 + dm1) with intial dm1=0

J1 = judiJacobian(F0_baseline, qb);  # J_x(m0+dm1)
J2 = judiJacobian(F0_monitor, qm);  # J_x(m0+dm2)

"""
Utility function that split the vector [m0; dm1; dm2] into each component (m0, dm1, dm2) for easy manipulation
"""
splitv(X) = X[1:N], X[N+1:2*N], X[2*N+1:end];

N = prod(n);

function TimeLapseLoss_jrm(X)
    # Split the components
    m0k, dm1, dm2 = splitv(X)
    # Random subset of source for each dataset
    binds, minds = randperm(nsrc)[1:batchsize], randperm(nsrc)[1:batchsize]
    # Synthetic data
    #d0b, d0m = F0_baseline(;m=m0k+dm1)[binds]*qb[binds], F0_monitor(;m=m0k+dm2)[minds]*qm[minds]
    d0b, d0m = F0_baseline(;m=m0k+dm1)[binds]*qb[binds], F0_monitor(;m=m0k+dm2)[binds]*qm[binds] #same seed
    # Set the propagation velocity to the current estimate
    J1.model.m .= m0k+dm1  # J_1(m0_k + dm1_k)
    J2.model.m .= m0k+dm2  # J_2(m0_k + dm1_k)
    # Compute the two gradient 
    g1 = J1[binds]'*(d0b - d_baseline[binds])
    #g2 = J2[minds]'*(d0m - d_monitor[minds])
    g2 = J2[binds]'*(d0m - d_monitor[binds]) #same seed
    
    # Mute water layer
    # While we haven't discussed this part, the water layer usually containts a very big imprint of 
    # the acquisition geometry making the inversion complicated. This could also be included as a constraint
    # but is usualy safer to always set it to zero here
    g1[:, 1:idx_wb] .= 0f0
    g2[:, 1:idx_wb] .= 0f0
    # Scale the gradient. In practice the amplitudes of the gradient are much different than the ones of the model
    # and rescaling it makes it easier for the optimization algorithm to converge. However we do not want to change
    # the expression of the misfit function so we set it as a constant based on the first gradient value at the first iteration
    g_const == 0 && (global g_const = .05f0 / norm(g1, Inf))
    g1 .*= g_const
    g2 .*= g_const
    # Misfit
    #phi = .5f0 * norm(d0b - d_baseline[binds])^2 + .5f0 * norm(d0m - d_monitor[minds])^2
    phi = .5f0 * norm(d0b - d_baseline[binds])^2 + .5f0 * norm(d0m - d_monitor[binds])^2 #same seed 
    
    # Set gradients of each component based on the matrix expression
    gm0 = g1[:]+g2[:]
    gdm1 = g1[:]
    gdm2 = g2[:]
    
    # We return the long vector [gm0;gdm1;gdm2] to match the input size since the optimization algorithm works best on pure vectors.
    return phi,  vcat(gm0, gdm1, gdm2)
end

function TimeLapseLoss_base(X)
    # Split the components
    m0k = X
    # Random subset of source for each dataset
    #inds = randperm(nsrc)[1:batchsize]
    inds = indices[iter]
    # Synthetic data
    d0b = F0_baseline(;m=m0k)[inds]*qb[inds]
    # Set the propagation velocity to the current estimate
    J1.model.m .= m0k  # J_1(m0_k + dm1_k)
    # Compute the two gradient 
    g1 = J1[inds]'*(d0b - d_baseline[inds])
    # Mute water layer
    # While we haven't discussed this part, the water layer usually containts a very big imprint of 
    # the acquisition geometry making the inversion complicated. This could also be included as a constraint
    # but is usualy safer to always set it to zero here
    g1[:, 1:idx_wb] .= 0f0
    # Scale the gradient. In practice the amplitudes of the gradient are much different than the ones of the model
    # and rescaling it makes it easier for the optimization algorithm to converge. However we do not want to change
    # the expression of the misfit function so we set it as a constant based on the first gradient value at the first iteration
    g_const == 0 && (global g_const = .05f0 / norm(g1, Inf))
    g1 .*= g_const
    # Misfit
    phi = .5f0 * norm(d0b - d_baseline[inds])^2
    # We return the long vector [gm0;gdm1;gdm2] to match the input size since the optimization algorithm works best on pure vectors.
    return phi,  g1
end

function TimeLapseLoss_mon1(X)
    # Split the components
    m0k = X
    # Random subset of source for each dataset
    #inds = randperm(nsrc)[1:batchsize]
    inds = indices[iter]
    # Synthetic data
    d0m = F0_monitor(;m=m0k)[inds]*qm[inds]
    # Set the propagation velocity to the current estimate
    J2.model.m .= m0k  # J_1(m0_k + dm1_k)
    # Compute the two gradient 
    g2 = J2[inds]'*(d0m - d_monitor[inds])
    # Mute water layer
    # While we haven't discussed this part, the water layer usually containts a very big imprint of 
    # the acquisition geometry making the inversion complicated. This could also be included as a constraint
    # but is usualy safer to always set it to zero here
    g2[:, 1:idx_wb] .= 0f0
    # Scale the gradient. In practice the amplitudes of the gradient are much different than the ones of the model
    # and rescaling it makes it easier for the optimization algorithm to converge. However we do not want to change
    # the expression of the misfit function so we set it as a constant based on the first gradient value at the first iteration
    g_const == 0 && (global g_const = .05f0 / norm(g2, Inf))
    g2 .*= g_const
    # Misfit
    phi = .5f0 * norm(d0m - d_monitor[inds])^2
    # We return the long vector [gm0;gdm1;gdm2] to match the input size since the optimization algorithm works best on pure vectors.
    return phi,  g2
end

first_callback = true
function callback(sol_jrm)
    if first_callback
        global first_callback = false
        return
    end
    global iter = iter+1
end

M0_jrm = vcat(m0[:], zeros(Float32, 2*N));
M0_ind = m0[:];
indices = GenSrcIdxLSRTM(nsrc, batchsize, niter);
# Check gradient
# phi0, G0 = TimeLapseLoss(M0)


# define constraints
cm0 = Vector{SetIntersectionProjection.set_definitions}();
c1 = Vector{SetIntersectionProjection.set_definitions}();

# Bound constraints on combined
m_min     = minimum(m_baseline);
m_max     = maximum(m_monitor);
set_type  = "bounds";
TD_OP     = "identity";
app_mode  = ("matrix","");
custom_TD_OP = ([],false);
push!(cm0, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP));

# Bound constraints for innovations
m_min     = -.02;
m_max     = .02;
set_type  = "bounds";
TD_OP     = "identity";
app_mode  = ("matrix","");
custom_TD_OP = ([],false);
push!(c1, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP));


# TV  constraint for innovations
TV = get_TD_operator(model_baseline, "TV", Float32)[1];
m_min     = 0.0;
m_max     = norm(TV*vec(m_baseline - m_monitor), 1); # Lets cheat for now and put the true TV
set_type  = "l1";
TD_OP     = "TV";
app_mode  = ("matrix","");
custom_TD_OP = ([],false);
push!(c1, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP));

#set up projectors
options=PARSDMM_options();
options.parallel = false;

(Pm0, TDm0, setm0) = setup_constraints(cm0, model_baseline, Float32);
(P1, TD1, set1) = setup_constraints(c1, model_baseline, Float32);

(TDm0, AtAm0, lm0, ym0) = PARSDMM_precompute_distribute(TDm0, setm0, model_baseline, options);
(TD1, AtA1, l1, y1) = PARSDMM_precompute_distribute(TD1, set1, model_baseline, options);

#set up one function to project the separate RGB channels of the image onto different constraint sets
Prm0 = x -> PARSDMM(x, AtAm0, TDm0, setm0, Pm0, model_baseline, options);
Pr1 = x -> PARSDMM(x, AtA1, TD1, set1, P1, model_baseline, options);

# Projection function for JRM recovery
function prj_jrm(input::T) where T
    input = Float32.(input)
    m0, dm1, dm2 = splitv(input)

    x1 = Prm0(m0 .+ dm1)[1]
    x2 = Prm0(m0 .+ dm2)[1]
    dx1 = Pr1(dm1)[1]
    dx2 = Pr1(dm2)[1]
    
    m0 = (x1 .+ x2 .- dx1 .- dx2) ./ 2

    return convert(T, vcat(m0, dx1, dx2))
end

# Projection function for independent recovery
function prj_ind(input::T) where T
    input = Float32.(input)

    x = Prm0(input)[1]

    return convert(T, x)
end

# FWI with SPG
iter = 1;
pqn_opt = pqn_options(verbose=3, maxIter=niter, memory=3, corrections=10);

sim_name = "sims_seis";
plot_path = plotsdir(sim_name);

if inv_type == "joint"
    println("*************** Running JRM recovery ****************")
    g_const = 0;
    sol_jrm = pqn(TimeLapseLoss_jrm, vec(M0_jrm), prj_jrm, pqn_opt; callback=callback)

    data_dict = @strdict nsrc nrec nv nv_id snr timeD dtD f0 niter batchsize inv_type space_order replicate sol_jrm 
    
    @tagsave(
        datadir(joinpath(sim_name, "sol_nrep_"*savename(data_dict, "jld2"; digits=6))),
        data_dict;
        safe=true)

    
    fig = figure(figsize=(12, 16))
    subplot(411)
    plot_velocity(reshape(splitv(sol_jrm.x)[1]+splitv(sol_jrm.x)[2], n)', d; new_fig=false, name="Baseline inverted model (nJRM)", cmap="cet_rainbow4_r", cbar=true)
    subplot(412)
    plot_velocity(reshape(splitv(sol_jrm.x)[1]+splitv(sol_jrm.x)[3], n)', d; new_fig=false, name="First monitor inverted model (nJRM)", cmap="cet_rainbow4_r", cbar=true)
    subplot(413)
    plot_simage(reshape(splitv(sol_jrm.x)[3]-splitv(sol_jrm.x)[2], n)', d; new_fig=false, vmax = 0.008, name="Inverted time-lapse (nJRM)", cmap="cet_CET_D1", cbar=true, perc=99.9)
    subplot(414)
    plot_simage(m_monitor' - m_baseline', d; new_fig=false, name="True time-lapse", vmax = 0.008, cmap="cet_CET_D1", cbar=true, perc=99)
    tight_layout()

    fig_name = @strdict nsrc nrec nv nv_id snr  timeD dtD f0 niter batchsize inv_type space_order replicate
    safesave(joinpath(plot_path, "sol_nrep_"*savename(fig_name; digits=6)*".png"), fig);

else
    println("*************** Running Independent recovery ****************")
    g_const = 0;
    sol_base = pqn(TimeLapseLoss_base, vec(M0_ind), prj_ind, pqn_opt; callback=callback)
    g_const = 0;
    first_callback = true;
    iter = 1;
    sol_mon1 = pqn(TimeLapseLoss_mon1, vec(M0_ind), prj_ind, pqn_opt; callback=callback)

    data_dict = @strdict nsrc nrec nv nv_id snr  timeD dtD f0 niter batchsize inv_type space_order replicate sol_base sol_mon1
    @tagsave(
        datadir(joinpath(sim_name, "sol_nrep_"*savename(data_dict, "jld2"; digits=6))),
        data_dict;
        safe=true)

    fig = figure(figsize=(12, 16))
    subplot(411)
    plot_velocity(reshape(sol_base.x, n)', d; new_fig=false, name="Baseline inverted model (Ind)", cmap="cet_rainbow4_r", cbar=true)
    subplot(412)
    plot_velocity(reshape(sol_mon1.x, n)', d; new_fig=false, name="First monitor inverted model (Ind)", cmap="cet_rainbow4_r", cbar=true)
    subplot(413)
    plot_simage(reshape(sol_mon1.x.-sol_base.x, n)', d; new_fig=false, vmax = 0.008, name="Inverted time-lapse (Ind)", cmap="cet_CET_D1", cbar=true, perc=99.9)
    subplot(414)
    plot_simage(m_monitor' - m_baseline', d; new_fig=false, vmax = 0.008, name="True time-lapse", cmap="cet_CET_D1", cbar=true, perc=99)
    tight_layout()

    fig_name = @strdict nsrc nrec nv nv_id snr  timeD dtD f0 niter batchsize inv_type space_order replicate 
    safesave(joinpath(plot_path, "sol_nrep_"*savename(fig_name; digits=6)*".png"), fig);
end 