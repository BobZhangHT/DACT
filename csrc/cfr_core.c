#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API
#endif

API int cfr_delay_exposure(
    int case_days, int outcome_days, int groups, int lags,
    const double *cases, const double *delay, double *exposure)
{
    if (case_days <= 0 || outcome_days <= 0 || groups <= 0 || lags <= 0)
        return -1;
    for (int i = 0; i < outcome_days * groups; ++i) exposure[i] = 0.0;
    for (int r = 0; r < case_days; ++r) {
        for (int j = 0; j < groups; ++j) {
            int stop = lags < outcome_days - r ? lags : outcome_days - r;
            for (int lag = 0; lag < stop; ++lag)
                exposure[(r + lag) * groups + j] +=
                    cases[r * groups + j] * delay[j * lags + lag];
        }
    }
    return 0;
}

/* Status: 0 interior convergence, 1 boundary, 2 exhausted iterations,
   3 nonfinite arithmetic, 4 no information, 5 no admissible split.
   Negative return codes are fatal allocation/input failures. */
static int valid_data(int weeks, int groups, const double *cases,
                      const double *deaths, const double *age)
{
    if (weeks <= 0 || groups <= 0) return 0;
    for (int j = 0; j < groups; ++j) if (!isfinite(age[j])) return 0;
    for (size_t i = 0; i < (size_t)weeks * groups; ++i)
        if (!isfinite(cases[i]) || !isfinite(deaths[i]) || cases[i] < 0.0 ||
            deaths[i] < 0.0 || (cases[i] == 0.0 && deaths[i] > 0.0)) return 0;
    return 1;
}

static double components(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double beta, double *score, double *info)
{
    double loglik = 0.0;
    for (int t = 0; t < weeks; ++t) {
        double max_eta = -INFINITY, total_deaths = 0.0;
        for (int j = 0; j < groups; ++j) {
            double eta = cases[t * groups + j] > 0.0 ?
                log(cases[t * groups + j]) + beta * age[j] : -INFINITY;
            if (eta > max_eta) max_eta = eta;
            total_deaths += deaths[t * groups + j];
        }
        if (total_deaths == 0.0) { score[t] = info[t] = 0.0; continue; }
        if (!isfinite(max_eta)) { score[t] = info[t] = NAN; return NAN; }
        double denominator = 0.0, mean_age = 0.0;
        for (int j = 0; j < groups; ++j) {
            if (cases[t * groups + j] == 0.0) continue;
            double eta = log(cases[t * groups + j]) + beta * age[j];
            denominator += exp(eta - max_eta);
        }
        for (int j = 0; j < groups; ++j) {
            if (cases[t * groups + j] == 0.0) continue;
            double eta = log(cases[t * groups + j]) + beta * age[j];
            double log_probability = eta - max_eta - log(denominator);
            double probability = exp(eta - max_eta) / denominator;
            mean_age += probability * age[j];
            if (deaths[t * groups + j] > 0.0)
                loglik += deaths[t * groups + j] * log_probability;
        }
        double observed_age = 0.0, variance_age = 0.0;
        for (int j = 0; j < groups; ++j) {
            if (cases[t * groups + j] == 0.0) continue;
            double eta = log(cases[t * groups + j]) + beta * age[j];
            double probability = exp(eta - max_eta) / denominator;
            observed_age += deaths[t * groups + j] * age[j];
            variance_age += probability * (age[j] - mean_age) * (age[j] - mean_age);
        }
        score[t] = observed_age - total_deaths * mean_age;
        info[t] = total_deaths * variance_age;
    }
    return loglik;
}

API double cfr_conditional_components(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double beta, double *score, double *info)
{
    if (!valid_data(weeks, groups, cases, deaths, age) || !isfinite(beta)) return NAN;
    return components(weeks, groups, cases, deaths, age, beta, score, info);
}

static double sum(int n, const double *x)
{
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += x[i];
    return total;
}

API int cfr_fit_beta(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double lower, double upper, double tolerance,
    int max_iterations, double *beta_out, double *loglik_out, int *iterations_out)
{
    *beta_out = *loglik_out = NAN; *iterations_out = 0;
    if (!valid_data(weeks, groups, cases, deaths, age) || !isfinite(lower) ||
        !isfinite(upper) || lower >= upper || !isfinite(tolerance) ||
        tolerance <= 0 || max_iterations <= 0) return -2;
    double *score = (double *)calloc((size_t)weeks, sizeof(double));
    double *info = (double *)calloc((size_t)weeks, sizeof(double));
    if (!score || !info) { free(score); free(info); return -1; }
    components(weeks, groups, cases, deaths, age, lower, score, info);
    double score_lower = sum(weeks, score);
    components(weeks, groups, cases, deaths, age, upper, score, info);
    double score_upper = sum(weeks, score);
    double beta = 0.5 * lower + 0.5 * upper;
    int status = 2;
    if (!isfinite(score_lower) || !isfinite(score_upper)) status = 3;
    else if (score_lower <= 0.0) { beta = lower; status = 1; }
    else if (score_upper >= 0.0) { beta = upper; status = 1; }
    else {
        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            *iterations_out = iteration;
            beta = 0.5 * lower + 0.5 * upper;
            components(weeks, groups, cases, deaths, age, beta, score, info);
            double value = sum(weeks, score);
            if (!isfinite(value)) { status = 3; break; }
            if (fabs(value) <= tolerance || upper - lower <= tolerance) { status = 0; break; }
            if (value > 0.0) lower = beta; else upper = beta;
        }
    }
    *loglik_out = components(weeks, groups, cases, deaths, age, beta, score, info);
    *beta_out = beta;
    double information = sum(weeks, info);
    if (!isfinite(*loglik_out) || !isfinite(sum(weeks, score)) || !isfinite(information)) status = 3;
    else if (information <= 0.0) status = 4;
    free(score); free(info);
    return status;
}

/* Cached score/information for the existing bracketed Newton batch solver. */
static int cached_components(int start, int end, int groups,
    const double *log_cases, const double *totals, const double *observed_age,
    const double *age, double beta, double *score, double *information)
{
    *score = *information = 0.0;
    for (int t = start; t < end; ++t) {
        if (totals[t] == 0.0) continue;
        double max_eta = -INFINITY, denominator = 0.0, mean = 0.0, second = 0.0;
        for (int j = 0; j < groups; ++j) {
            double eta = log_cases[t * groups + j] + beta * age[j];
            if (eta > max_eta) max_eta = eta;
        }
        if (!isfinite(max_eta)) return 3;
        for (int j = 0; j < groups; ++j) {
            double weight = exp(log_cases[t * groups + j] + beta * age[j] - max_eta);
            denominator += weight; mean += weight * age[j]; second += weight * age[j] * age[j];
        }
        mean /= denominator; second /= denominator;
        *score += observed_age[t] - totals[t] * mean;
        *information += totals[t] * (second - mean * mean);
    }
    return isfinite(*score) && isfinite(*information) ? 0 : 3;
}

static int cached_fit(int start, int end, int groups,
    const double *log_cases, const double *totals, const double *observed_age,
    const double *age, double lower, double upper, double tolerance,
    int max_iterations, double initial, double *beta_out)
{
    double value, information;
    *beta_out = NAN;
    if (cached_components(start, end, groups, log_cases, totals, observed_age,
                          age, lower, &value, &information)) return 3;
    if (value <= 0.0) { *beta_out = lower; return information <= 0.0 ? 4 : 1; }
    if (cached_components(start, end, groups, log_cases, totals, observed_age,
                          age, upper, &value, &information)) return 3;
    if (value >= 0.0) { *beta_out = upper; return information <= 0.0 ? 4 : 1; }
    double beta = initial > lower && initial < upper ? initial : 0.5 * lower + 0.5 * upper;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        *beta_out = beta;
        if (cached_components(start, end, groups, log_cases, totals, observed_age,
                              age, beta, &value, &information)) return 3;
        if (information <= 0.0) return 4;
        if (fabs(value) <= tolerance || upper - lower <= tolerance) return 0;
        if (value > 0.0) lower = beta; else upper = beta;
        double candidate = beta + value / fmax(information, 1e-14);
        beta = candidate > lower && candidate < upper ? candidate : 0.5 * lower + 0.5 * upper;
    }
    return 2;
}

API int cfr_bootstrap_change_batch(
    int samples, int weeks, int groups, const double *cases,
    const double *sampled_deaths, const double *age, double trim_fraction, double lower,
    double upper, double tolerance, int max_iterations, int *changes,
    double *beta_before, double *beta_after, int *statuses)
{
    if (samples <= 0 || weeks <= 0 || groups <= 0 || !isfinite(lower) || !isfinite(upper) ||
        lower >= upper || !isfinite(tolerance) || tolerance <= 0 || max_iterations <= 0 ||
        !isfinite(trim_fraction) || trim_fraction < 0 || trim_fraction >= 0.5) return -2;
    double *log_cases = (double *)malloc((size_t)weeks * groups * sizeof(double));
    double *observed_age = (double *)malloc((size_t)weeks * sizeof(double));
    double *totals = (double *)malloc((size_t)weeks * sizeof(double));
    double *daily_score = (double *)malloc((size_t)weeks * sizeof(double));
    double *daily_info = (double *)malloc((size_t)weeks * sizeof(double));
    if (!log_cases || !observed_age || !totals || !daily_score || !daily_info) {
        free(log_cases); free(observed_age); free(totals);
        free(daily_score); free(daily_info); return -1;
    }
    for (size_t i = 0; i < (size_t)weeks * groups; ++i)
        log_cases[i] = cases[i] > 0.0 ? log(cases[i]) : -INFINITY;
    const size_t stride = (size_t)weeks * groups;
    for (int b = 0; b < samples; ++b) {
        const double *deaths = sampled_deaths + (size_t)b * stride;
        changes[b] = 0; beta_before[b] = beta_after[b] = NAN; statuses[b] = 3;
        if (!valid_data(weeks, groups, cases, deaths, age)) continue;
        for (int t = 0; t < weeks; ++t) {
            observed_age[t] = totals[t] = 0.0;
            for (int j = 0; j < groups; ++j) {
                observed_age[t] += deaths[t * groups + j] * age[j];
                totals[t] += deaths[t * groups + j];
            }
        }
        /* Use the same null-root solve as the public fit: tiny Newton/root
           differences can select different calendar days on tied sparse paths. */
        double null_beta, null_loglik;
        int null_iterations;
        statuses[b] = cfr_fit_beta(weeks, groups, cases, deaths, age, lower, upper,
                                  tolerance, max_iterations, &null_beta, &null_loglik, &null_iterations);
        if (statuses[b]) continue;
        double loglik = components(weeks, groups, cases, deaths, age, null_beta, daily_score, daily_info);
        double total_info = sum(weeks, daily_info);
        if (!isfinite(loglik) || !isfinite(total_info) || !isfinite(sum(weeks, daily_score))) { statuses[b] = 3; continue; }
        if (total_info <= 0.0) { statuses[b] = 4; continue; }
        double cumulative = 0.0, cumulative_info = 0.0, largest = -1.0;
        int change = 0;
        for (int t = 0; t < weeks - 1; ++t) {
            cumulative += daily_score[t]; cumulative_info += daily_info[t];
            double fraction = cumulative_info / total_info;
            if (fraction < trim_fraction || fraction > 1.0 - trim_fraction) continue;
            double value = fabs(cumulative);
            if (value > largest) { largest = value; change = t + 1; }
        }
        if (!change) { statuses[b] = 5; continue; }
        changes[b] = change;
        for (int segment = 0; segment < 2; ++segment) {
            int start = segment == 0 ? 0 : change, end = segment == 0 ? change : weeks;
            double beta;
            statuses[b] = cached_fit(start, end, groups, log_cases, totals, observed_age, age,
                                    lower, upper, tolerance, max_iterations, null_beta, &beta);
            if (statuses[b]) break;
            loglik = components(end - start, groups, cases + start * groups, deaths + start * groups,
                                age, beta, daily_score, daily_info);
            total_info = sum(end - start, daily_info);
            if (!isfinite(loglik) || !isfinite(total_info) || !isfinite(sum(end - start, daily_score))) {
                statuses[b] = 3; break;
            }
            if (total_info <= 0.0) { statuses[b] = 4; break; }
            if (segment == 0) beta_before[b] = beta; else beta_after[b] = beta;
        }
    }
    free(log_cases); free(observed_age); free(totals); free(daily_score); free(daily_info);
    return 0;
}
