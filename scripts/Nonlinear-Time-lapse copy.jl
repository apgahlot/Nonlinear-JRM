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
ENV["OMP_NUM_THREADS"] = div(Sys.CPU_THREADS, 2)

using SlimPlotting, JUDI, JLD2, JOLI, SlimOptim, SegyIO, SetIntersectionProjection
using PyPlot, Images, Random, LinearAlgebra, StatsBase, DSP
using ArgParse
matplotlib.use("agg")
Random.seed!(2023)

include(srcdir("Seis4CCSops.jl"))
include(srcdir("utils.jl"))

parsed_args = parse_commandline()
nv = parsed_args["nv"]
nsrc = parsed_args["nsrc"]
nrec = parsed_args["nrec"]
#idx = parsed_args["idx"]
snr = parsed_args["snr"]
niter = parsed_args["niter"]
batchsize = parsed_args["batchsize"]
inv_type = parsed_args["inv_type"]
#inv_type = "independent"


d = (15.0f0, 15.0f0)
o = (0f0, 0f0)

# Get Data
data_dict = @strdict nsrc nrec nv snr
if isfile(datadir("dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6)))
    @load datadir("dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6)) d_baseline qb
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
    data_dict = @strdict nsrc nrec nv snr d_baseline qb
    @tagsave(
        datadir("dbaseline_nrep_"*savename(data_dict, "jld2"; digits=6)),
        data_dict;
        safe=true)
end

if isfile(datadir("dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6)))
    @load datadir("dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6)) d_monitor qm
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
    data_dict = @strdict nsrc nrec nv snr d_monitor qm
    @tagsave(
        datadir("dmonitor_nrep_"*savename(data_dict, "jld2"; digits=6)),
        data_dict;
        safe=true)
end

# v0 = deepcopy(v1)
# v0[:,idx_wb+1:end] = 1f0./imfilter(1f0./v1[:,idx_wb+1:end], Kernel.gaussian(10))
# m0 = (1f0./v0).^2f0
# rho0 = deepcopy(rho1)
# rho0[:,idx_wb+1:end] = 1f0./imfilter(1f0./rho1[:,idx_wb+1:end], Kernel.gaussian(10))
# dimp = [vec(v0.*rho0-v1.*rho1), vec(v0.*rho0-v2.*rho2)]

m_baseline = convert(Matrix{Float32}, m_stack[1][1:4:end, 1:2:end])
m_monitor = convert(Matrix{Float32}, m_stack[3][1:4:end, 1:2:end])
idx_wb = maximum(find_water_bottom(m_baseline .- maximum(m_baseline)))
wb = find_water_bottom(m_baseline .- maximum(m_baseline))
m0 = 1 .* m_baseline;
m0[:, idx_wb:end] = imfilter(m0[:, idx_wb:end].^(.5), Kernel.gaussian(5f0)).^2;
m0 = convert(Matrix{Float32}, m0)

# Model size
n = size(m_baseline)

# Acquisition parameters
timeD = 3000f0  # We record the pressure at the receivers for 3000ms (3 Sec) for each source
dtD = 2f0   # We sample the measurements every 2 ms
f0 = 0.0145f0  # We use a 14.5Hz source (0.0145 KHz since we measure time in ms) and band pass it to a realitic frequency band
wavelet = low_filter(ricker_wavelet(timeD, dtD, f0), 4f0; fmin=3, fmax=15);

# Source coordinates in physical units (m)
#xsrc = [convertToCell(ContJitter((n[1]-1)*d[1], nsrc)), convertToCell(ContJitter((n[1]-1)*d[1], nsrc))]
qb = qb
qm = qm
xsrc1 = [vcat(qb[i].geometry.xloc...) for i = 1:d_baseline.nsrc]
xsrc2 = [vcat(qm[i].geometry.xloc...) for i = 1:d_monitor.nsrc]
xsrc = [xsrc1, xsrc2];
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc)) # Set y coordinate to zero since it's 2D
zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc))  # Sources at 15m depth

# JUDI geometry object
#srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
srcGeometry_b = Geometry(xsrc[1], ysrc, zsrc; dt=dtD, t=timeD)
srcGeometry_m = Geometry(xsrc[2], ysrc, zsrc; dt=dtD, t=timeD)

# Receivers (measurments) coordinates in phyisical units
#xrec = range(start=100f0, step=100f0, stop=(n[1]-1)*d[1]-100f0)  # Receivers every 100m
xrec = d_baseline.geometry.xloc  # Receivers every 120m
yrec = d_baseline.geometry.yloc # WE have to set the y coordinate to zero (or any number) for 2D modeling
#zrec = (wb[5:4:end-5] .- 1) .* d[2]
zrec = d_baseline.geometry.zloc

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

model_baseline = Model(n, d, o, m_baseline; nb=80)
model_monitor = Model(n, d, o, m_monitor; nb=80)

opt = Options(dt_comp=1f0) # We fix the computational time-step to avoid slight variations in numerical scheme leading to differences in the data.
F_baseline = judiModeling(model_baseline, srcGeometry_b, recGeometry; options=opt)
F_monitor = judiModeling(model_monitor, srcGeometry_m, recGeometry; options=opt)

# Define the operator for the background
F0_b = F_baseline(;m=m0)  # F(m0 + dm1) with intial dm1=0
F0_m = F_monitor(;m=m0) # F(m0 + dm1) with intial dm1=0

J1 = judiJacobian(F0_b, qb)  # J_x(m0+dm1)
J2 = judiJacobian(F0_m, qm)  # J_x(m0+dm2)

"""
Utility function that split the vector [m0; dm1; dm2] into each component (m0, dm1, dm2) for easy manipulation
"""
splitv(X) = X[1:N], X[N+1:2*N], X[2*N+1:end]

N = prod(n)

"""
Main objective function. This is the function that, for a given input [m0; dm1; dm2] computes:
- the misfit value (the \ell_2 norm of each data difference)
- The three gradient using the simplified expression that only requires the application of J1 and J2 once.
"""
function TimeLapseLoss(X)
    # Split the components
    m0k, dm1, dm2 = splitv(X)
    # Random subset of source for each dataset
    binds, minds = randperm(nsrc)[1:batchsize], randperm(nsrc)[1:batchsize]
    # Synthetic data
    #d0b, d0m = F0_b(;m=m0k+dm1)[binds]*qb[binds], F0_m(;m=m0k+dm2)[minds]*qm[minds]
    d0b, d0m = F0_b(;m=m0k+dm1)[binds]*qb[binds], F0_m(;m=m0k+dm2)[binds]*qm[binds] #same seed
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
    g1[:, 1:12] .= 0f0
    g2[:, 1:12] .= 0f0
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
    gdm1 = g1
    gdm2 = g2
    
    # We return the long vector [gm0;gdm1;gdm2] to match the input size since the optimization algorithm works best on pure vectors.
    return phi,  vcat(gm0, gdm1, gdm2)
end

first_callback = true
function callback(sol)
    if first_callback
        global first_callback = false
        return
    end
    global iter = iter+1
end

M0 = vcat(m0[:], zeros(Float32, 2*N));

# define constraints
cm0 = Vector{SetIntersectionProjection.set_definitions}()
c1 = Vector{SetIntersectionProjection.set_definitions}()

# Bound constraints on combined
m_min     = minimum(m_monitor)
m_max     = maximum(m_baseline)
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(cm0, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP))

# Bound constraints for innovations
m_min     = -.001
m_max     = .001
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(c1, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP))


# TV  constraint for innovations
TV = get_TD_operator(model_baseline, "TV", Float32)[1]
m_min     = 0.0
m_max     = norm(TV*vec(m_baseline - m_monitor), 1) # Lets cheat for now and put the true TV
m_max     = norm(TV*vec(m_baseline - m_monitor), 1) # Lets cheat for now and put the true TV
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(c1, set_definitions(set_type, TD_OP, m_min, m_max, app_mode, custom_TD_OP))

#set up projectors
options=PARSDMM_options()
options.parallel = false

(Pm0, TDm0, setm0) = setup_constraints(cm0, model_baseline, Float32);
(P1, TD1, set1) = setup_constraints(c1, model_baseline, Float32);

(TDm0, AtAm0, lm0, ym0) = PARSDMM_precompute_distribute(TDm0, setm0, model_baseline, options)
(TD1, AtA1, l1, y1) = PARSDMM_precompute_distribute(TD1, set1, model_baseline, options)

#set up one function to project the separate RGB channels of the image onto different constraint sets
Prm0 = x -> PARSDMM(x, AtAm0, TDm0, setm0, Pm0, model_baseline, options)
Pr1 = x -> PARSDMM(x, AtA1, TD1, set1, P1, model_baseline, options)

# Projection function
function prj(input::T) where T
    input = Float32.(input)
    m0, dm1, dm2 = splitv(input)

    x1 = Prm0(m0 .+ dm1)[1]
    x2 = Prm0(m0 .+ dm2)[1]
#     dx1 = Pr1(dm1)[1]
#     dx2 = Pr1(dm2)[1]
    dx1 = Prm0(dm1)[1]
    dx2 = Prm0(dm2)[1]
    
    m0 = (x1 .+ x2 .- dx1 .- dx2) ./ 2

    return convert(T, vcat(m0, dx1, dx2))
end

g_const = 0
iter = 1
# FWI with SPG
pqn_opt = pqn_options(verbose=3, maxIter=niter, memory=3, corrections=10)
sol_jrm = pqn(TimeLapseLoss, vec(M0), prj, pqn_opt; callback=callback)

data_dict = @strdict nsrc nrec nv snr sol_jrm
@tagsave(
    datadir("sol_jrm_nrep_"*savename(data_dict, "jld2"; digits=6)),
    data_dict;
    safe=true)

sim_name = "sims_image"
plot_path = plotsdir(sim_name)

figure(figsize=(12, 16))
subplot(411)
plot_velocity(reshape(splitv(sol.x)[1]+splitv(sol.x)[2], n)', d; new_fig=false, name="Baseline inverted model", cmap="cet_rainbow4_r", cbar=true)
subplot(412)
plot_simage(reshape(splitv(sol.x)[3]-splitv(sol.x)[2], n)', d; new_fig=false, name="Inverted time-lapse", cmap="cet_CET_D1", cbar=true, perc=99.9)
subplot(413)
plot_simage(m_monitor' - m_baseline', d; new_fig=false, name="True time-lapse", cmap="cet_CET_D1", cbar=true, perc=99)
subplot(414)
plot_velocity(reshape(splitv(sol.x)[1]+splitv(sol.x)[3], n)', d; new_fig=false, name="First vintage inverted model", cmap="cet_rainbow4_r", cbar=true)
tight_layout()

fig_name = @strdict type nsrc nrec nv snr
safesave(joinpath(plot_path, savename(fig_name; digits=6)*".png"), fig);
