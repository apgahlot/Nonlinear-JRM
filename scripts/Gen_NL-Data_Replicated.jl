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
dt = parsed_args["dt"]
nt = parsed_args["nt"]
irate = parsed_args["irate"]
timeD = parsed_args["timeD"]
dtD = parsed_args["dtD"]
f0 = parsed_args["f0"]
replicate = parsed_args["replicate"]

data_path = "sims_jutul"
data_dict = @strdict dt nt irate 
@load datadir(joinpath(data_path, "velrho_10m_nJRM_"*savename(data_dict, "jld2"; digits=6))) v_stack rho_stack

v1, v2 = v_stack[1][1:4:end, 1:2:end], v_stack[nv_id][1:4:end, 1:2:end]
#rho1, rho2 = rho_stack[1][1:2:end, 1:2:end], rho_stack[(nv_id-1)*5][1:2:end, 1:2:end]
# Model size
n = size(v1)
d = (15.0f0, 15.0f0)
o = (0f0, 0f0)

idx_wb = maximum(find_water_bottom(v1.-v1[1,1]))

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

m_baseline = (1f0./v1).^2f0
m_monitor = (1f0./v2).^2f0
m0 = 1 .* m_baseline;
m0[:, idx_wb:end] = imfilter(m0[:, idx_wb:end].^(.5), Kernel.gaussian(5f0)).^2;
m0 = convert(Matrix{Float32}, m0)

# Acquisition parameters
timeD = timeD  # We record the pressure at the receivers for 3000ms (3 Sec) for each source
dtD = dtD   # We sample the measurements every 2 ms
f0 = f0  # We use a 14.5Hz source (0.0145 KHz since we measure time in ms) and band pass it to a realitic frequency band
wavelet = filter_data(ricker_wavelet(timeD, dtD, f0), 4f0; fmin=3, fmax=15);

# Source coordinates in physical units (m)
#xsrc = [convertToCell(ContJitter((n[1]-1)*d[1], nsrc)), convertToCell(ContJitter((n[1]-1)*d[1], nsrc))]
xsrc = convertToCell(range(start=25f0, stop=(n[1]-1)*d[1]-30f0, length=nsrc)) # Sources are every 39m at the surface.
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc)) # Set y coordinate to zero since it's 2D
zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc))  # Sources at 15m depth

# JUDI geometry object
#srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
srcGeometry_b = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
srcGeometry_m = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

# Receivers (measurments) coordinates in phyisical units
#xrec = range(start=100f0, step=100f0, stop=(n[1]-1)*d[1]-100f0)  # Receivers every 100m
xrec = range(start=50f0, stop=(n[1]-1)*d[1]-50f0,length=nrec) 
yrec = 0f0 # WE have to set the y coordinate to zero (or any number) for 2D modeling
zrec = range((idx_wb-1)*d[2],stop=(idx_wb-1)*d[2],length=nrec)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

qb = judiVector(srcGeometry_b, wavelet)
qm = judiVector(srcGeometry_m, wavelet)

model_baseline = Model(n, d, o, m_baseline; nb=80)
model_monitor = Model(n, d, o, m_monitor; nb=80)

space_order = 16
opt = Options(dt_comp=1f0, space_order=space_order) # We fix the computational time-step to avoid slight variations in numerical scheme leading to differences in the data.
F_baseline = judiModeling(model_baseline, srcGeometry_b, recGeometry; options=opt)
F_monitor = judiModeling(model_monitor, srcGeometry_m, recGeometry; options=opt)

# Data generation
data_dict = @strdict nsrc nrec nv nv_id snr timeD dtD f0 space_order replicate
sim_name = "sims_seis"
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

plot_path = plotsdir(sim_name)
fig = figure(figsize=(20, 20))
for i=1:5
    sxi = (i + 1) * div(nsrc, 7)
    sxb = xsrc[sxi][1]
    sxm = xsrc[sxi][1]
    subplot(3,5,i)
    plot_sdata(d_baseline.data[sxi],d; cmap="PuOr", new_fig=false, name="Baseline sx=$(sxb)m", cbar=true)
    subplot(3,5,i+5)
    plot_sdata(d_monitor.data[sxi],d; cmap="PuOr", new_fig=false, name="Monitor sx=$(sxm)m", cbar=true)
    subplot(3,5,i+10)
    plot_sdata(d_baseline.data[sxi] - d_monitor.data[sxi],d; cmap="PuOr", new_fig=false, name="Time-lapse sx=$(sxb)m", cbar=true)
end
tight_layout()

fig_name = @strdict nsrc nrec nv nv_id snr timeD dtD f0 space_order replicate
safesave(joinpath(plot_path, "Shot_record_"*savename(fig_name; digits=6)*".png"), fig)
