clear
close all
load('steady.mat')   % loads: grid, p, sstate (from KSsteady.m)

tic;

% Step size for numerical differentiation of prices (r, w).
% Scaled to wage to keep perturbation size economically meaningful.
epsh = [sstate.w sstate.w]' * 10^-4;

T = 300;          % Length of the time horizon for the sequence-space Jacobian
nprices = size(sstate.prices, 1);   % Number of prices being perturbed (r and w)


% Pre-allocate policy and distribution objects
cpol  = zeros(grid.nk, grid.nz);
Ppol  = sparse(grid.nk*grid.nz, grid.nk*grid.nz);
kpol  = zeros(grid.nk, grid.nz, nprices);
Vk_f  = zeros(size(cpol));          % Perturbed marginal value of capital (function of prices only, not distribution)
Ds    = zeros(grid.nk*grid.nz, 1);  % Deviation of joint distribution from steady state

% FK(t, s, pp): Fake News Matrix entry — effect on aggregate capital at
% date t of news at date s about price pp arriving at date T
FK = zeros(T, T, nprices);

% Convenience: vectorised steady-state capital policy and reshaped joint distribution
kl             = repmat(grid.kgrid, grid.nz, 1);
sstate.kdashl  = sstate.kpol(:);                       % steady-state k' as a column vector
sstate.JDr     = reshape(sstate.JD, [grid.nk, grid.nz]); % joint distribution on (k,z) grid


%% Calculate Fake News Matrix
% Algorithm: Proposition 2 of Auclert et al. (2021).
% For each news horizon s, we:
%   1. Solve backwards from T to T-s to find how the policy function
%      responds to a one-time price perturbation announced s periods ahead.
%   2. Compute the on-impact effect (row 1 of FK) from the policy deviation.
%   3. Propagate the distribution deviation forward (rows 2:T of FK)
%      using the steady-state transition matrix Ph.

for t = 0:(T-1)     % t indexes how far back from T the news arrives; s = t+1 is the column of FK

    tt = T - t;     % current backward step: we are solving period tt of the T-period sequence

    for pp = 1:nprices

        prices = sstate.prices;

        if tt == T
            % At the furthest-back date (period T), initialise with the
            % steady-state marginal value and apply the price perturbation.
            Vk_f(:,:,pp)  = sstate.Vk;
            prices(pp)    = prices(pp) + epsh(pp);
        end

        % One backward step: update policy and transition matrix under perturbed prices.
        % Vk_f is updated in place as we step backwards toward period 1.
        [cpol, kpol, Vk_f(:,:,pp)] = polupdate(prices, Vk_f(:,:,pp), p, grid, sstate);
        [Ppol]                      = tranupdate(kpol, grid, sstate);

        % --- On-impact row of the Fake News Matrix (t=0, i.e. row 1) ---
        % Measures the immediate change in aggregate K caused by the policy
        % deviation, weighted by the steady-state distribution.
        s = t + 1;
        FK(1, s, pp) = sum(sum(sstate.JDr .* (kpol - sstate.kpol))) / epsh(pp);

        % Ds: first-order change in the joint distribution one period after
        % the policy deviation, relative to the steady-state distribution.
        Ds = (sstate.JD' * Ppol)' - sstate.JD;

        % --- Forward propagation rows of the Fake News Matrix (t >= 1) ---
        % After the initial shock, prices return to steady state but the
        % distribution continues to evolve. We propagate Ds forward using
        % Ph (the steady-state transition matrix) and record how each
        % shifted distribution affects aggregate K.
        for ttt = 2:T
            FK(ttt, s, pp) = sum(Ds .* sstate.kdashl) / epsh(pp);
            Ds = (Ds' * sstate.Ph)';   % one period forward under steady-state transitions
        end

    end
end


%% Recover the Sequence-Space Jacobian from the Fake News Matrix
% Proposition 1 of Auclert et al. (2021):
%   J(t,s) = F(t,s) + J(t-1,s-1)
% i.e. the Jacobian cumulates the fake news effects along diagonals.
% The first row and column are identical to FK (boundary condition).

JK = zeros(T, T, nprices);

for pp = 1:nprices
    JK(1, :, pp) = FK(1, :, pp);   % on-impact row
    JK(:, 1, pp) = FK(:, 1, pp);   % contemporaneous shock column
end

for tt = 2:T
    for pp = 1:nprices
        JK(tt, 2:end, pp) = JK(tt-1, 1:end-1, pp) + FK(tt, 2:end, pp);
    end
end

toc;

%% Replicate Figure 2 of Auclert et al. (2021)
% Jacobian and Fake News Matrix for K with respect to r

JKr = JK(:,:,1);   % Jacobian of K w.r.t. r
FKr = FK(:,:,1);   % Fake News Matrix of K w.r.t. r

JKw = JK(:,:,2);   % Jacobian of K w.r.t. w  (used below for GE)
FKw = FK(:,:,2);

figure(1)
clf

subplot(3,1,1)
plot(JKr(:,[1,25,50,75,100]))
legend(string([1,25,50,75,100]))
title('Jacobian for K at different horizon shocks to r')

subplot(3,1,2)
plot(FKr(:,1))
legend(string(1))
title('Fake news matrix for initial shock')

subplot(3,1,3)
plot(FKr(:,[25,50,75,100]))
legend(string([25,50,75,100]))
title('Fake news paths for shocks at different horizons')


%% Impulse Response Function: K response to a TFP shock
% General equilibrium condition (in sequence space):
%   F(x, Z) = 0  where  K' = G(x, Z)
%   x = [r, w, K],  Z = TFP
%
% Linearising around steady state gives:
%   Fk * dx + Fz * dZ = 0   =>   dx = -Fk^{-1} * Fz * dZ
%
% Fk and Fz are constructed by chaining the partial Jacobians of prices
% w.r.t. capital/TFP through the sequence-space Jacobians JKr, JKw.

% Derivatives of factor prices w.r.t. aggregate capital (from Cobb-Douglas)
dwdk = (1-p.aalpha) * p.aalpha  * sstate.Kd^(p.aalpha-1) * sstate.N^(-p.aalpha);
drdk = p.aalpha * (p.aalpha-1)  * sstate.Kd^(p.aalpha-2) * sstate.N^(1-p.aalpha);

% Derivatives of factor prices w.r.t. TFP
drdz = p.aalpha       * sstate.Kd^(p.aalpha-1) * sstate.N^(1-p.aalpha);
dwdz = (1-p.aalpha)   * sstate.Kd^(p.aalpha)   * sstate.N^(-p.aalpha);

% Jacobian of market-clearing condition w.r.t. K (T-1 x T-1 system)
% Subtract identity to impose K_t = K_{t+1} in equilibrium
Fk = JKr*drdk + JKw*dwdk;
Fk = Fk(2:end, 1:end-1) - eye(T-1, T-1);

% Jacobian of market-clearing condition w.r.t. TFP path
Fz = JKr*drdz + JKw*dwdz;
Fz = Fz(1:end-1, 1:end-1);

% AR(1) TFP shock path with persistence rho = 0.9
dz = 1 * power(0.90*ones(T-1,1), (0:T-2)');

% Solve for equilibrium capital path: dx = -Fk^{-1} * Fz * dz
dx = -1 * (Fk \ eye(T-1,T-1)) * Fz * dz;

figure(2)
clf
plot(dx(1:50))
title('Response of K to a TFP shock')
