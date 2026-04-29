"""Rewrite the benchmark cells in (final)cont+single.ipynb with correct Eq. (15) logic."""
import json

with open('(final)cont+single.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [src]}

# =====================================================================
# Cell 87: hazard rate helper — unchanged, correct
# =====================================================================
C87 = md("## Hazard Rate & Virtual Index (Pavan et al. 2014, Eq. 15)")

C88 = code(
    "import jax.scipy.stats as stats\n"
    "\n"
    "# Type distribution parameters (must match create_bandit_auction_two_items)\n"
    "MU_PRIOR    = 2.5\n"
    "SIGMA_PRIOR = 0.1\n"
    "\n"
    "def hazard_rate_fn(x, mu=MU_PRIOR, sigma=SIGMA_PRIOR):\n"
    "    \"\"\"Hazard rate h(x) = f(x) / (1 - F(x)) of the initial Gaussian prior.\n"
    "\n"
    "    Used as 1/h(theta_i0) in the virtual index formula (PST 2014, Eq. 15).\n"
    "    The hazard rate is always evaluated at the INITIAL type theta_i0,\n"
    "    not at the current type, because the IC rent is determined by theta_i0.\n"
    "    \"\"\"\n"
    "    f = stats.norm.pdf(x, loc=mu, scale=sigma)\n"
    "    F = stats.norm.cdf(x, loc=mu, scale=sigma)\n"
    "    return f / jnp.maximum(1.0 - F, 1e-12)\n"
)

# =====================================================================
# Cell 89: explanation of Eq. 15
# =====================================================================
C89 = md(
    "## Eq. (15): Stopping-Time Virtual Index\n\n"
    "PST 2014 Equation 15 defines the **virtual index** for buyer i with *initial* type\n"
    "$\\theta_{i0}$ and *current* type $\\theta_{it}$ as the Gittins-like stopping-time problem:\n\n"
    "$$\\mu_i(\\theta_{it},\\, \\theta_{i0})\n"
    "= \\sup_{\\tau \\ge t}\n"
    "\\frac{E\\!\\left[\\sum_{s=t}^{\\tau} \\delta^{s-t}\\left(\\theta_{is}\n"
    "- \\tfrac{1-F_0(\\theta_{i0})}{f_0(\\theta_{i0})}\\right)\n"
    "\\Bigg| \\theta_{it},\\; x_{is}=1 \\text{ for } s \\le \\tau\\right]}\n"
    "{E\\!\\left[\\sum_{s=t}^{\\tau} \\delta^{s-t}\\right]}$$\n\n"
    "Key points:\n"
    "- The **inverse hazard rate** $1/h_0(\\theta_{i0}) = (1-F_0(\\theta_{i0}))/f_0(\\theta_{i0})$\n"
    "  is evaluated at the **initial type $\\theta_{i0}$** and held *fixed* throughout\n"
    "  (it captures the IC rent, not the current-period info).\n"
    "- The *expectation* conditions on the **current type $\\theta_{it}$** and always\n"
    "  allocating to buyer i up to $\\tau$ (so the type evolves via the *allocated* Markov chain).\n"
    "- The supremum over $\\tau$ gives the optimal stopping time.\n\n"
    "The **allocation rule**: at each period t, allocate to\n"
    "$i^* = \\operatorname{argmax}_i(\\mu_i - c_i)$ if this is $\\ge 0$.\n\n"
    "The **revenue formula** follows from PST Theorem 2:\n"
    "$E[\\text{Rev}] = E[\\sum_t \\delta^t \\sum_i (\\theta_{it} - 1/h_0(\\theta_{i0}) - c_i)\\, x_{it}]$"
)

# =====================================================================
# Cell 90: correct compute_indices (fixes: state reset, sum→mean)
# =====================================================================
C90 = code(
    "def compute_indices(auction: BanditAuction, state: State,\n"
    "                    inv_hazard_rates: jnp.ndarray,\n"
    "                    num_samples: int, key: jax.random.PRNGKey):\n"
    "    \"\"\"Compute virtual indices for each buyer via PST 2014, Eq. (15).\n"
    "\n"
    "    mu_i(theta_it, theta_i0) = max_{tau >= t}\n"
    "        E[ sum_{s=t}^{tau} delta^{s-t} * (theta_is - inv_hazard_i) | theta_it, allocated ]\n"
    "        / E[ sum_{s=t}^{tau} delta^{s-t} ]\n"
    "\n"
    "    inv_hazard_rates[i] = (1 - F0(theta_i0)) / f0(theta_i0) is FIXED at the\n"
    "    initial type theta_i0 for the entire game (passed in from simulate_benchmark).\n"
    "\n"
    "    Bugs fixed vs. original:\n"
    "    - State is now reset fresh for every (buyer, tau_stop, sample) triple,\n"
    "      so samples are independent and tau iterations don't bleed into each other.\n"
    "    - Expected value is taken as mean over samples (was incorrectly summed).\n"
    "    - allocation_cost is NOT included in the index numerator\n"
    "      (it is subtracted at the allocation-decision step instead).\n"
    "    \"\"\"\n"
    "    t = int(state.time_step)\n"
    "    T = auction.num_horizons\n"
    "    gamma = float(auction.discount)\n"
    "    B = auction.num_buyers\n"
    "    sample_keys = jax.random.split(key, num_samples)\n"
    "\n"
    "    virtual_indices = []\n"
    "    for i in range(B):\n"
    "        inv_hz_i = float(inv_hazard_rates[i])  # 1/h0(theta_i0), fixed\n"
    "\n"
    "        # --- Draw num_samples paths of length (T-t) by always allocating to buyer i.\n"
    "        # Each path shares the same starting state (state) but uses an independent key.\n"
    "        all_paths = []   # all_paths[k] = list of theta_is for s in [t, T)\n"
    "        for sk in sample_keys:\n"
    "            sim = state  # Python rebind; JAX struct is immutable, so this is a fresh ref\n"
    "            path = []\n"
    "            for s in range(t, T):\n"
    "                sk, step_key = jax.random.split(sk)\n"
    "                path.append(float(sim.current_type[i, 0]))\n"
    "                # always allocate to buyer i\n"
    "                alloc = jnp.zeros((B, auction.num_goods))\n"
    "                alloc = jax.lax.dynamic_update_index_in_dim(\n"
    "                    alloc, jnp.array([1.0]), i, axis=0)\n"
    "                sim = auction.get_next_state(sim, sim.current_type, alloc, step_key)\n"
    "            all_paths.append(path)\n"
    "\n"
    "        # --- For each candidate stopping time, compute the average ratio.\n"
    "        best_ratio = float('-inf')\n"
    "        for tau_stop in range(t, T):\n"
    "            length = tau_stop - t + 1\n"
    "            ratios = []\n"
    "            for path in all_paths:\n"
    "                num = sum(\n"
    "                    gamma ** s * (path[s] - inv_hz_i)\n"
    "                    for s in range(length)\n"
    "                )\n"
    "                den = sum(gamma ** s for s in range(length))\n"
    "                ratios.append(num / den)\n"
    "            mean_ratio = float(np.mean(ratios))\n"
    "            if mean_ratio > best_ratio:\n"
    "                best_ratio = mean_ratio\n"
    "\n"
    "        virtual_indices.append(best_ratio)\n"
    "\n"
    "    return jnp.array(virtual_indices)\n"
    "\n"
    "\n"
    "def compute_alloc(auction: BanditAuction, state: State,\n"
    "                  inv_hazard_rates: jnp.ndarray,\n"
    "                  num_samples: int, key: jax.random.PRNGKey):\n"
    "    \"\"\"Allocate to the buyer with highest non-negative (virtual index - cost).\"\"\"\n"
    "    indices = compute_indices(auction, state, inv_hazard_rates, num_samples, key)\n"
    "    net = indices - auction.allocation_cost[:, 0]   # subtract c_i at decision stage\n"
    "    best = int(jnp.argmax(net))\n"
    "    allocation = jnp.zeros((auction.num_buyers, auction.num_goods))\n"
    "    if float(net[best]) >= 0.0:\n"
    "        allocation = jax.lax.dynamic_update_index_in_dim(\n"
    "            allocation, jnp.array([1.0]), best, axis=0)\n"
    "    return best, allocation\n"
)

# =====================================================================
# Cell 91: simulate benchmark (corrected)
# =====================================================================
C91 = md(
    "## Simulate Benchmark\n\n"
    "Revenue formula (PST 2014, Theorem 2):\n\n"
    "$$E[\\text{Rev}] = E\\!\\left[\\sum_{t=0}^{T-1}\\delta^t \\sum_i\n"
    "(\\theta_{it} - 1/h_0(\\theta_{i0}) - c_i)\\,x_{it}\\right]$$\n\n"
    "The inverse hazard $1/h_0(\\theta_{i0})$ is computed **once at $t=0$** from the\n"
    "initial type and kept fixed for the whole episode (it does not update with\n"
    "the current type $\\theta_{it}$)."
)

C92 = code(
    "def simulate_benchmark_pavan(auction: BanditAuction, num_epochs: int,\n"
    "                             num_samples: int, key: jax.random.PRNGKey):\n"
    "    \"\"\"Simulate the PST 2014 index policy and estimate expected revenue.\n"
    "\n"
    "    Follows Eq. (15) exactly:\n"
    "    - inv_hazard_rates are computed at the INITIAL type theta_i0 and held fixed.\n"
    "    - At each period t, compute_indices uses the CURRENT types theta_it but the\n"
    "      FIXED initial inverse hazard rates.\n"
    "    - Revenue is accumulated as sum_t delta^t * (theta_it - 1/h0 - c_i) * x_it.\n"
    "    \"\"\"\n"
    "    total_revenues = []\n"
    "    epoch_keys = jax.random.split(key, num_epochs)\n"
    "\n"
    "    for epoch in range(num_epochs):\n"
    "        ep_key = epoch_keys[epoch]\n"
    "        state = auction.get_initial_state(ep_key)\n"
    "\n"
    "        # --- Inverse hazard rates: fixed at initial type theta_i0 for this episode.\n"
    "        hazard_rates = jnp.array([\n"
    "            hazard_rate_fn(state.current_type[i, 0])\n"
    "            for i in range(auction.num_buyers)\n"
    "        ])\n"
    "        inv_hazard_rates = 1.0 / hazard_rates   # (1-F0)/f0, fixed for whole episode\n"
    "\n"
    "        ep_revenue = 0.0\n"
    "        for t in range(auction.num_horizons):\n"
    "            ep_key, alloc_key = jax.random.split(ep_key)\n"
    "\n"
    "            # Allocation based on virtual indices (current types, initial hazard)\n"
    "            winner, allocation = compute_alloc(\n"
    "                auction, state, inv_hazard_rates, num_samples, alloc_key\n"
    "            )\n"
    "\n"
    "            # Virtual surplus = (theta_it - 1/h0_i - c_i) * x_it, per buyer\n"
    "            virtual_surplus = jnp.sum(\n"
    "                (state.current_type[:, 0] - inv_hazard_rates\n"
    "                 - auction.allocation_cost[:, 0]) * allocation[:, 0]\n"
    "            )\n"
    "            ep_revenue += (auction.discount ** t) * float(virtual_surplus)\n"
    "\n"
    "            ep_key, trans_key = jax.random.split(ep_key)\n"
    "            state = auction.get_next_state(\n"
    "                state, state.current_type, allocation, trans_key\n"
    "            )\n"
    "\n"
    "        total_revenues.append(ep_revenue)\n"
    "        if (epoch + 1) % 200 == 0:\n"
    "            print(f'  Epoch {epoch+1}/{num_epochs}: '\n"
    "                  f'mean={np.mean(total_revenues):.4f} '\n"
    "                  f'std={np.std(total_revenues):.4f}')\n"
    "\n"
    "    return total_revenues\n"
)

C93 = code(
    "key = jax.random.PRNGKey(SEED)\n"
    "total_revenues, = simulate_benchmark_pavan(auction, num_epochs=1000,\n"
    "                                            num_samples=50, key=key),\n"
    "mu_rev  = np.mean(total_revenues)\n"
    "std_rev = np.std(total_revenues)\n"
    "print(f'Benchmark revenue: {mu_rev:.4f} +/- {std_rev:.4f}')\n"
)

# =====================================================================
# Splice: replace cells 87-95 (9 current corrected cells) with new 7
# =====================================================================
new_cells = [C87, C88, C89, C90, C91, C92, C93]

# Current state: cells 87-95 were our previous (incorrect) correction
# Replace them with the new correct version
nb['cells'] = nb['cells'][:87] + new_cells + nb['cells'][96:]

with open('(final)cont+single.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
