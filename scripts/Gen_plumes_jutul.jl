using DrWatson
@quickactivate "Nonlinear-JRM"

using  JutulDarcyRules, JUDI, JOLI, JLD2, Polynomials, Random
using  PyPlot, SlimPlotting, Images
using ArgParse
matplotlib.use("agg")

include(srcdir("Seis4CCSops.jl"))
include(srcdir("utils.jl"))

parsed_args = parse_commandline()
dt = parsed_args["dt"]
nt = parsed_args["nt"]
irate = parsed_args["irate"]

#@load datadir("BGCompass_tti_625m.jld2") n d o m rho
@load datadir("CompassSlice_Image23.jld2") n d o m rho

vp = sqrt.(1f0./m);

function VtoK(vp::Matrix{T}, d::Tuple{T, T}; α::T=T(20)) where T

    n = size(vp)
    idx_wb = find_water_bottom(vp.-minimum(vp))
    idx_ucfmt = find_water_bottom((vp.-T(3.5)).*(vp.>T(3.5)))
    Kh = zeros(T, n)
    capgrid = Int(round(T(50)/d[2]))
    for i = 1:n[1]
        Kh[i,1:idx_wb[i]-1] .= T(1e-10)  # water layer
        Kh[i,idx_wb[i]:idx_ucfmt[i]-capgrid-1] .= α*exp.(vp[i,idx_wb[i]:idx_ucfmt[i]-capgrid-1])
    # Create an opening in the seal
    if i > 1030 && i < 1060
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(2e1)
    elseif i > 1061 && i < 1081
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(1e1)
    elseif i > 1082 && i < 1092
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(5e0)
    elseif i > 1093 && i < 1098
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(1e0)
    else
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(1e-3)
    end
        Kh[i,idx_ucfmt[i]:end] .= α*exp.(vp[i,idx_ucfmt[i]:end]) .- α*exp(T(3.5))
        # This is to remove some erratic perm values at the boundary between seal and reservoir
        if Kh[i,idx_ucfmt[i]] < T(10)
            Kh[i,idx_ucfmt[i]] = T(10)
        end
    end
    return Kh
end

K = VtoK(vp, d);
K[:,1:maximum(idx_wb)] .= 0.0; # correction for Water layer

phi = zeros(Float64,n[1],n[end]);
idx_wb = find_water_bottom(vp[:,:].-minimum(vp[:,:]));
idx_ucfmt = find_water_bottom((vp[:,:].-3.5).*(vp[:,:].>3.5));
capgrid = Int(round(50/d[2]))
for i = 1:n[1]
    for j = 1:n[end]
        p = Polynomial([-0.0314^2*K[i,j],2*0.0314^2*K[i,j],-0.0314^2*K[i,j],1.527^2]) 
        phi[i,j] = minimum(real(roots(p)[findall(real(roots(p)).== roots(p))]))
    end
    for j = idx_ucfmt[i]:idx_ucfmt[i]-capgrid-1
        phi[i,idx_ucfmt[i]:idx_ucfmt[i]-capgrid-1] = Float64.(range(0.056,stop=0.1,length=capgrid+1))
    end
end
phi[:,1:maximum(idx_wb)] .= 1.0 # correction for Water layer

# Jutul 
Kmd = Float64.(K * md);
## grid size
n = (size(Kmd,1), 1, size(Kmd,2));
d = Float64.((d[1], 50.0, d[2]));

ϕ = convert(Array{Float64,1},vec(phi));

model = jutulModel(n, d, ϕ, K1to3(Kmd))

## simulation time steppings
dt = dt;
nt = nt;
tstep = dt * ones(nt);
tot_time = sum(tstep);

## injection & production
#Kmax_loc = 271 + argmax(Kmd[220,271:280]) - 1;
Kmax_loc = 160 + argmax(Kmd[550,160:170]) - 1;
#inj_loc = (220, 1, Kmax_loc) .* d;
inj_loc = (550, 1, Kmax_loc) .* d;
#prod_loc = (580, 1, Kmax_loc) .* d;
prod_loc = (1400, 1, Kmax_loc) .* d;
irate = irate
#q = jutulVWell(irate, (inj_loc[1], inj_loc[2]); startz = inj_loc[3], endz = inj_loc[3]+37.5)

q = jutulVWell(irate, [(inj_loc[1],inj_loc[2]), (prod_loc[1],prod_loc[2])],
               startz = [inj_loc[3],inj_loc[3]], endz = [inj_loc[3]+37.5,inj_loc[3]+37.5]);

#q = jutulForce(irate, [inj_loc, prod_loc])

## set up modeling operator
S = jutulModeling(model, tstep);

## simulation
Trans = KtoTrans(CartesianMesh(model), K1to3(Kmd; kvoverkh=0.66));
@time state = S(log.(Trans), q, info_level=1)

states25_nJRM = vcat(Saturations(state),Pressure(state));

sat_nJRM = zeros(Float64,nt+1,n[1],n[end]);
for i = 2:nt+1
    sat_nJRM[i,:,:] = reshape(Saturations(state.states[i-1]), (n[1], n[end]))
end

v_stack, rho_stack = Patchy(sat_nJRM[:,:,:], Float64.(vp), Float64.(rho), 
                            phi,(d[1],d[end]));

v_stack = Matrix{Float32}.(v_stack);
m_stack = [1.0f0./v_stack[i].^2 for i = 1:length(v_stack)];
rho_stack = Matrix{Float32}.(rho_stack);

sim_name = "sims_jutul"
data_dict = @strdict n d dt nt inj_loc prod_loc irate states25_nJRM v_stack rho_stack
@tagsave(
    datadir(joinpath(sim_name, "velrho_10m_nJRM_"*savename(data_dict, "jld2"; digits=6))),
    data_dict;
    safe=true
)

plot_path = plotsdir(sim_name)
fig = figure(figsize=(16,16))
subplot(511)
plot_velocity(reshape(Saturations(state.states[2]), (n[1], n[end]))',(d[1],d[end]),name="CO2 Saturation", new_fig=false, cbar=true)
subplot(512)
plot_velocity(reshape(Saturations(state.states[4]), (n[1], n[end]))',(d[1],d[end]),name="CO2 Saturation", new_fig=false, cbar=true)
subplot(513)
plot_velocity(reshape(Saturations(state.states[6]), (n[1], n[end]))',(d[1],d[end]),name="CO2 Saturation", new_fig=false, cbar=true)
subplot(514)
plot_velocity(reshape(Saturations(state.states[8]), (n[1], n[end]))',(d[1],d[end]),name="CO2 Saturation", new_fig=false, cbar=true)
subplot(515)
plot_velocity(reshape(Saturations(state.states[10]), (n[1], n[end]))',(d[1],d[end]),name="CO2 Saturation", new_fig=false, cbar=true)
tight_layout()
fig_name = @strdict n d dt nt irate
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_CO2_saturation.png"), fig);

fig = figure(figsize=(16,16))
subplot(511)
plot_velocity(v_stack[1]',(d[1],d[end]),name="Baseline Velocity (Km/s)", new_fig=false, cbar=true)
subplot(512)
plot_velocity(v_stack[end]',(d[1],d[end]),name="Monitor Velocity (Km/s)", new_fig=false, cbar=true)
subplot(513)
plot_velocity(rho_stack[1]',(d[1],d[end]),name="Baseline Density (g/cc)", new_fig=false, cbar=true)
subplot(514)
plot_velocity(rho_stack[end]',(d[1],d[end]),name="Monitor Density (g/cc)", new_fig=false, cbar=true)
subplot(515)
plot_velocity(v_stack[1]' .- v_stack[end]',(d[1],d[end]),name="Velocity Time-lapse change (Km/s)", new_fig=false, cbar=true)
tight_layout()
fig_name = @strdict n d dt nt irate
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_Time-lapse_change_velrho.png"), fig);


fig = figure(figsize=(16,16))
subplot(511)
plot_velocity(m_stack[3]' .- m_stack[1]',(d[1],d[end]),name="Slowness Difference first vintage", new_fig=false, cbar=true)
subplot(512)
plot_velocity(m_stack[5]' .- m_stack[1]',(d[1],d[end]),name="Slowness Difference second vintage", new_fig=false, cbar=true)
subplot(513)
plot_velocity(m_stack[7]' .- m_stack[1]',(d[1],d[end]),name="Slowness Difference third vintage", new_fig=false, cbar=true)
subplot(514)
plot_velocity(m_stack[9]' .- m_stack[1]',(d[1],d[end]),name="Slowness Difference fourth vintage", new_fig=false, cbar=true)
subplot(515)
plot_velocity(m_stack[11]' .- m_stack[1]',(d[1],d[end]),name="Slowness Difference fifth vintage", new_fig=false, cbar=true)
tight_layout()
fig_name = @strdict n d dt nt irate
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_Time-lapse_change_slowness.png"), fig);
