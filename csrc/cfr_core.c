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

static double components(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double beta, double *score, double *info)
{
    double loglik = 0.0;
    for (int t = 0; t < weeks; ++t) {
        double max_eta = -INFINITY, total_deaths = 0.0;
        for (int j = 0; j < groups; ++j) {
            double eta = log(fmax(cases[t * groups + j], 1e-12)) + beta * age[j];
            if (eta > max_eta) max_eta = eta;
            total_deaths += deaths[t * groups + j];
        }
        double denominator = 0.0, mean_age = 0.0;
        for (int j = 0; j < groups; ++j) {
            double eta = log(fmax(cases[t * groups + j], 1e-12)) + beta * age[j];
            denominator += exp(eta - max_eta);
        }
        for (int j = 0; j < groups; ++j) {
            double eta = log(fmax(cases[t * groups + j], 1e-12)) + beta * age[j];
            double probability = exp(eta - max_eta) / denominator;
            mean_age += probability * age[j];
            if (deaths[t * groups + j] > 0.0)
                loglik += deaths[t * groups + j] * log(probability);
        }
        double observed_age = 0.0, variance_age = 0.0;
        for (int j = 0; j < groups; ++j) {
            double eta = log(fmax(cases[t * groups + j], 1e-12)) + beta * age[j];
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
    return components(weeks, groups, cases, deaths, age, beta, score, info);
}

API int cfr_fit_beta(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double lower, double upper, double tolerance,
    int max_iterations, double *beta_out, double *loglik_out, int *iterations_out)
{
    double *score = (double *)malloc((size_t)weeks * sizeof(double));
    double *info = (double *)malloc((size_t)weeks * sizeof(double));
    if (!score || !info) { free(score); free(info); return -1; }
    components(weeks, groups, cases, deaths, age, lower, score, info);
    double score_lower = 0.0;
    for (int t = 0; t < weeks; ++t) score_lower += score[t];
    components(weeks, groups, cases, deaths, age, upper, score, info);
    double score_upper = 0.0;
    for (int t = 0; t < weeks; ++t) score_upper += score[t];
    double beta;
    int iterations = 0;
    if (score_lower <= 0.0) beta = lower;
    else if (score_upper >= 0.0) beta = upper;
    else {
        for (iterations = 0; iterations < max_iterations; ++iterations) {
            beta = 0.5 * (lower + upper);
            components(weeks, groups, cases, deaths, age, beta, score, info);
            double value = 0.0;
            for (int t = 0; t < weeks; ++t) value += score[t];
            if (fabs(value) <= tolerance || upper - lower <= tolerance) break;
            if (value > 0.0) lower = beta; else upper = beta;
        }
        beta = 0.5 * (lower + upper);
    }
    *loglik_out = components(weeks, groups, cases, deaths, age, beta, score, info);
    *beta_out = beta;
    *iterations_out = iterations + 1;
    free(score); free(info);
    return 0;
}

static int fit_beta_with_buffers(
    int weeks, int groups, const double *cases, const double *deaths,
    const double *age, double lower_bound, double upper_bound,
    double tolerance, int max_iterations, double *score, double *info,
    double *beta_out)
{
    double lower = lower_bound, upper = upper_bound;
    components(weeks, groups, cases, deaths, age, lower, score, info);
    double score_lower = 0.0;
    for (int t = 0; t < weeks; ++t) score_lower += score[t];
    components(weeks, groups, cases, deaths, age, upper, score, info);
    double score_upper = 0.0;
    for (int t = 0; t < weeks; ++t) score_upper += score[t];
    double beta = 0.5 * (lower + upper);
    if (score_lower <= 0.0) beta = lower;
    else if (score_upper >= 0.0) beta = upper;
    else {
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            beta = 0.5 * (lower + upper);
            components(weeks, groups, cases, deaths, age, beta, score, info);
            double value = 0.0;
            for (int t = 0; t < weeks; ++t) value += score[t];
            if (fabs(value) <= tolerance || upper - lower <= tolerance) break;
            if (value > 0.0) lower = beta; else upper = beta;
        }
        beta = 0.5 * (lower + upper);
    }
    components(weeks, groups, cases, deaths, age, beta, score, info);
    *beta_out = beta;
    return 0;
}

API int cfr_bootstrap_change_batch(
    int samples, int weeks, int groups, const double *cases,
    const double *sampled_deaths, const double *age, double lower,
    double upper, double tolerance, int max_iterations, int *changes,
    double *beta_before, double *beta_after)
{
    double *log_cases = (double *)malloc((size_t)weeks * groups * sizeof(double));
    double *observed_age = (double *)malloc((size_t)weeks * sizeof(double));
    double *totals = (double *)malloc((size_t)weeks * sizeof(double));
    if (!log_cases || !observed_age || !totals) {
        free(log_cases); free(observed_age); free(totals); return -1;
    }
    for (int i = 0; i < weeks * groups; ++i)
        log_cases[i] = log(fmax(cases[i], 1e-12));
    const size_t stride = (size_t)weeks * (size_t)groups;
    for (int b = 0; b < samples; ++b) {
        const double *deaths = sampled_deaths + (size_t)b * stride;
        for (int t = 0; t < weeks; ++t) {
            observed_age[t] = 0.0;
            totals[t] = 0.0;
            for (int j = 0; j < groups; ++j) {
                observed_age[t] += deaths[t * groups + j] * age[j];
                totals[t] += deaths[t * groups + j];
            }
        }

        /* A bracketed Newton solve uses the same score equation as
           cfr_fit_beta, but avoids likelihood and per-week array work. */
        double lo = lower, hi = upper, null_beta = 0.5 * (lo + hi);
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double total_score = 0.0, total_info = 0.0;
            for (int t = 0; t < weeks; ++t) {
                double max_eta = -INFINITY, denominator = 0.0;
                double mean = 0.0, second = 0.0;
                for (int j = 0; j < groups; ++j) {
                    double eta = log_cases[t * groups + j] + null_beta * age[j];
                    if (eta > max_eta) max_eta = eta;
                }
                for (int j = 0; j < groups; ++j) {
                    double weight = exp(log_cases[t * groups + j] + null_beta * age[j] - max_eta);
                    denominator += weight; mean += weight * age[j];
                    second += weight * age[j] * age[j];
                }
                mean /= denominator; second /= denominator;
                total_score += observed_age[t] - totals[t] * mean;
                total_info += totals[t] * (second - mean * mean);
            }
            if (fabs(total_score) <= tolerance || hi - lo <= tolerance) break;
            if (total_score > 0.0) lo = null_beta; else hi = null_beta;
            double candidate = null_beta + total_score / fmax(total_info, 1e-14);
            null_beta = (candidate > lo && candidate < hi) ? candidate : 0.5 * (lo + hi);
        }
        double cumulative = 0.0, largest = -1.0;
        int change = 1;
        for (int t = 0; t < weeks - 1; ++t) {
            double max_eta = -INFINITY, denominator = 0.0, mean = 0.0;
            for (int j = 0; j < groups; ++j) {
                double eta = log_cases[t * groups + j] + null_beta * age[j];
                if (eta > max_eta) max_eta = eta;
            }
            for (int j = 0; j < groups; ++j) {
                double weight = exp(log_cases[t * groups + j] + null_beta * age[j] - max_eta);
                denominator += weight; mean += weight * age[j];
            }
            cumulative += observed_age[t] - totals[t] * mean / denominator;
            double value = fabs(cumulative);
            if (value > largest) { largest = value; change = t + 1; }
        }
        changes[b] = change;
        for (int segment = 0; segment < 2; ++segment) {
            int start = segment == 0 ? 0 : change;
            int end = segment == 0 ? change : weeks;
            lo = lower; hi = upper;
            double beta = null_beta;
            if (beta <= lo || beta >= hi) beta = 0.5 * (lo + hi);
            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                double total_score = 0.0, total_info = 0.0;
                for (int t = start; t < end; ++t) {
                    double max_eta = -INFINITY, denominator = 0.0;
                    double mean = 0.0, second = 0.0;
                    for (int j = 0; j < groups; ++j) {
                        double eta = log_cases[t * groups + j] + beta * age[j];
                        if (eta > max_eta) max_eta = eta;
                    }
                    for (int j = 0; j < groups; ++j) {
                        double weight = exp(log_cases[t * groups + j] + beta * age[j] - max_eta);
                        denominator += weight; mean += weight * age[j];
                        second += weight * age[j] * age[j];
                    }
                    mean /= denominator; second /= denominator;
                    total_score += observed_age[t] - totals[t] * mean;
                    total_info += totals[t] * (second - mean * mean);
                }
                if (fabs(total_score) <= tolerance || hi - lo <= tolerance) break;
                if (total_score > 0.0) lo = beta; else hi = beta;
                double candidate = beta + total_score / fmax(total_info, 1e-14);
                beta = (candidate > lo && candidate < hi) ? candidate : 0.5 * (lo + hi);
            }
            if (segment == 0) beta_before[b] = beta; else beta_after[b] = beta;
        }
    }
    free(log_cases);
    free(observed_age);
    free(totals);
    return 0;
}
