import numpy as np
import os
import george
import requests
import scipy.optimize as op
from astropy.table import Table
from astroquery.gaia import Gaia

path = "nearby-sample.fits"

if not os.path.exists(path):

    job = Gaia.launch_job("""
        SELECT  *,
                phot_g_mean_mag + 5 * log10(parallax/100.) as absolute_g_mag,
                sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al - 5)) as astrometric_unit_weight_error 
        FROM    gaiadr2.gaia_source 
        WHERE   parallax >= 50
            AND duplicated_source = 'false'
            AND phot_g_mean_mag + 5 * log10(parallax/100.) < (6.5 + 2*bp_rp)  
            AND phot_g_mean_mag > 6
        """)

    sample = job.get_results()
    sample.write(path, format="fits")

else:
    sample = Table.read(path)



# Gaussian process model

x = np.array([
    sample["bp_rp"],
    sample["absolute_g_mag"],
    sample["phot_g_mean_mag"]
]).T
y = np.array(sample["astrometric_unit_weight_error"])


metric = np.var(x, axis=0)
kernel = george.kernels.ExpSquaredKernel(metric, ndim=x.shape[1])

gp = george.GP(kernel, 
               mean=np.mean(y), fit_mean=True,
               white_noise=np.log(np.std(y)), fit_white_noise=True)

def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

"""
# This is hack week, after all.
is_outlier = y > 5
yerr = np.zeros(y.size)
yerr[is_outlier] = 50

gp.compute(x, yerr=yerr)

print("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
print("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

p0 = gp.get_parameter_vector()

result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

gp.set_parameter_vector(result.x)

print("Result: {}".format(result))
print("Final logL = {:.2f}".format(gp.log_likelihood(y)))

p, p_var = gp.predict(y, x, return_var=True)

import matplotlib.pyplot as plt

kwds = dict(s=5, vmin=0, vmax=7)
fig, ax = plt.subplots(1, 2)
ax[0].scatter(x.T[0], x.T[1], c=p, **kwds)
ax[0].set_ylim(ax[0].get_ylim()[::-1])

ax[1].scatter(x.T[0], x.T[1], c=y, **kwds)
ax[1].set_ylim(ax[1].get_ylim()[::-1])


fig, ax = plt.subplots()
ax.scatter(y, p, s=5)
plt.show()
"""


def optimize_given_hyper_hyper_parameters(theta):

    a, b = theta

    yerr = np.zeros(y.size)
    yerr[y > a] = b

    gp.compute(x, yerr=yerr)
    p0 = gp.get_parameter_vector()
    result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    gp.set_parameter_vector(result.x)

    z = -gp.log_likelihood(y)
    print(theta, z)
    return z 


#p0 = [5, 50]
#op.minimize(optimize_given_hyper_hyper_parameters, p0, method="Nelder-Mead")

# These parameters were found by optimizing from [5, 50]
optimize_given_hyper_hyper_parameters([3.78727253, 19.26386123])


p, p_var = gp.predict(y, x, return_var=True)

import matplotlib.pyplot as plt

kwds = dict(s=5, vmin=0, vmax=7)
fig, ax = plt.subplots(1, 2)
ax[0].scatter(x.T[0], x.T[1], c=p, **kwds)
ax[0].set_ylim(ax[0].get_ylim()[::-1])

ax[0].set_xlabel("bp - rp")
ax[0].set_ylabel("absolute g mag")


scat = ax[1].scatter(x.T[0], x.T[1], c=y, **kwds)
ax[1].set_ylim(ax[1].get_ylim()[::-1])

ax[1].set_xlabel("bp - rp")
ax[1].set_ylabel("absolute g mag")

ax[0].set_title("single star model")
ax[1].set_title("data")

cbar = plt.colorbar(scat)
cbar.set_label("astrometric unit weight error [clipped]")

fig.tight_layout()

fig.savefig("nearby-sample.pdf", dpi=300)



fig, ax = plt.subplots()
ax.scatter(y, p, s=5)
ax.errorbar(y, p, yerr=p_var**0.5, fmt='none')
plt.show()




excess_sigma = (y - p)/np.sqrt(p_var)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

ax[0].scatter(x.T[0], x.T[1], c=y, s=5, vmin=0, vmax=7)

ax[1].scatter(x.T[0], x.T[1], s=5, c=excess_sigma, vmin=0, vmax=10)

ax[1].set_ylim(ax[1].get_ylim()[::-1])


sample["predicted_auwe"] = p
sample["predicted_auwe_error"] = p_var**0.5


# Select 39 sources and do Simbad query.
is_candidate = (excess_sigma > 10) * (sample["bp_rp"] < 2)

substrings = [
    "binary",
    "hierarchy"
]

for substring in substrings:
    sample[f"simbad_{substring}"] = np.zeros(len(sample), dtype=bool)


indices = np.where(is_candidate)[0]

for index in indices:
    #candidate in sample[is_candidate]:

    candidate = sample[index]

    foo = requests.get(
        "http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=Gaia+DR2+{0}&submit=SIMBAD+search".format(
        candidate["source_id"]))

    for substring in substrings:

        try:
            foo.text.index(substring)

        except ValueError:
            continue

        else:
            sample[f"simbad_{substring}"][index] = True
            print(f"matched {substring}")
    

# Save the candidate file.
sample.write(path, format="fits", overwrite=True)
sample[is_candidate].write("nearby-sample-candidates.fits", format="fits", overwrite=True)

# Print an output.
print("#source_id,ra,dec,simbad_binary,simbad_hierarchy,simbad_link")
for candidate in sample[is_candidate]:
    print(f"{candidate['source_id']:>20},{candidate['ra']:10.5f},{candidate['dec']:10.5f},{candidate['simbad_binary']},{candidate['simbad_hierarchy']},http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=Gaia+DR2+{candidate['source_id']}&submit=SIMBAD+search")



