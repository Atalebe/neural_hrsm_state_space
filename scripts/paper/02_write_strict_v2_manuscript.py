#!/usr/bin/env python3
"""
Write the strict-v2 Neural HRSM manuscript source.

This manuscript version incorporates:
    - six strict fixed-scale sessions
    - explicit neural H/R/S/M formulae
    - raw versus orthogonalized axis audit
    - six-session target synthesis
    - six-session variance/autocorrelation residual audit
    - six-session lag-ablation audit
    - six-session ripeness summary
"""

from pathlib import Path


TEX = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{natbib}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{hyperref}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!50!black,
    citecolor=blue!50!black,
    urlcolor=blue!50!black
}

\graphicspath{{./}{../}}
\input{paper/tables/strict_v2_macros.tex}

\title{Controlled Organizational Memory in Spontaneous Neural Population Dynamics:\\
A Six-Session HRSM State-Space Analysis of Allen Neuropixels Data}

\author{Stephen Atalebe\\
\emph{Affiliation to be confirmed}}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Spontaneous neural activity is often treated either as background variability or as an internally structured dynamical process. This study asks whether spontaneous neural population states contain a retained historical component that improves prediction of future population organization. Six strict fixed-scale Allen Visual Coding Neuropixels spontaneous sessions were analyzed using a reduced homeostatic state-space model with four operational coordinates: \(H\), \(R\), \(S\), and \(M\). Each strict session used four target regions, twenty units per region, fifteen spontaneous presentations, and matched binning. For each session, population activity was binned, projected into HRSM state space, and tested using aligned lag-history predictors against shuffled-lag controls. Across the strict cohort, active-unit fraction and population-rate entropy were the strongest cross-session controlled-memory targets, with median observed-minus-shuffled gains of \StrictActiveMedianGain{} and \StrictEntropyMedianGain{}, respectively. Variance and autocorrelation residualization preserved these two targets as the leading residualized targets. Lag-ablation tests showed that both recruitment and entropy remained positive across all tested lag settings, with controlled gain increasing across a short recent-history window. Session-level heterogeneity remained, including one strict stress-test session in which mean-rate memory was not uniformly above shuffled controls. These results support a bounded conclusion: spontaneous neural population activity carries controlled organizational memory, most visible in population recruitment and entropy, without implying consciousness, awareness, or a complete theory of neural dynamics.
\end{abstract}

\noindent\textbf{Keywords:} spontaneous neural activity; Neuropixels; Allen Brain Observatory; population dynamics; non-Markovian memory; entropy; recruitment; HRSM; state-space analysis.

\section{Introduction}

Neural population activity is commonly represented as an instantaneous state vector. Such a representation is necessary, but it may be incomplete. If two neural populations occupy similar current states while differing in recent history, their future organization may still diverge. The present work asks whether such reduced-state incompleteness can be detected operationally in spontaneous neural activity. The test is statistical: does aligned recent population history improve prediction of future population organization relative to a present-state baseline and relative to a shuffled-lag null?

Spontaneous neural activity is not empty baseline. Large-scale recordings have shown that spontaneous activity can be high-dimensional, structured, and related to ongoing behavior or internal state \citep{Stringer2019Spontaneous}. Work on latent neural states has emphasized that neural variability is shaped by non-stationary internal dynamics, behavior, and stimulus context \citep{Akella2025LatentStates}. Hidden-state analyses of spontaneous spiking activity similarly indicate that ongoing population activity can be organized into latent dynamical regimes \citep{Sederberg2024LatentDynamics}. The present study is aligned with this literature, but it focuses on a narrower question: whether retained recent history contributes to prediction of future spontaneous population organization.

The analysis uses the Allen Visual Coding Neuropixels dataset, which surveys spiking activity across cortical, thalamic, hippocampal, and related visual-system structures using high-density Neuropixels probes \citep{Jun2017Neuropixels,Siegle2021VisualHierarchy}. The Allen Brain Observatory provides standardized, reusable neurophysiology datasets and has become an important resource for computational analyses of visual coding and neural population dynamics \citep{DeVries2023AllenLessons,Xie2023VisualCodingAssessment}. Here, the stimulus-driven components of the dataset are not the primary focus. Instead, spontaneous intervals are used to test whether ongoing population dynamics carry controlled historical structure.

This manuscript deliberately uses bounded language. The result is not presented as independent external replication, because all sessions come from the Allen Visual Coding Neuropixels dataset. It is within-dataset cross-session generalization across six strict fixed-scale sessions. The result is also not presented as universal mean-rate memory. The strongest cross-session evidence concerns organizational targets, especially active-unit fraction and population-rate entropy.

\section{Data and preprocessing}

\subsection{Strict fixed-scale session cohort}

Six Allen Visual Coding Neuropixels sessions were included in the strict fixed-scale cohort:
\[
715093703,\quad
719161530,\quad
750749662,\quad
751348571,\quad
755434585,\quad
756029989.
\]
Each strict session contributed four target regions:
\[
\mathrm{VISp},\quad \mathrm{VISl},\quad \mathrm{LGd},\quad \mathrm{CA1}.
\]
Each region contributed twenty selected units. Each session used fifteen spontaneous presentations, a 0.25 second bin size, and a maximum duration of 60 seconds per spontaneous presentation. Session 754312389 was excluded from the strict cohort and retained only as a partial-coverage diagnostic because VISl contributed fewer than twenty units.

\begin{table}[t]
\centering
\caption{Strict six-session extraction summary. Each strict session used four regions, twenty units per region, fifteen spontaneous presentations, and matched binning.}
\label{tab:strict-extraction}
\input{paper/tables/table_strict_v2_extraction_summary.tex}
\end{table}

The analysis used direct HDF5 access to cached NWB files rather than constructing full PyNWB session objects. This was a computational choice intended to reduce memory pressure while preserving access to spike times, unit identifiers, electrode-region metadata, and spontaneous-presentation intervals.

\subsection{Population-state matrix}

For each session and region, binned spike counts were converted into a population-state matrix. Each row represented a time-bin population state within a spontaneous presentation. The main target variables were population mean rate, population rate standard deviation, active-unit fraction, population L2 rate norm, population-rate entropy, and population-state speed. Active-unit fraction quantifies recruitment. Population-rate entropy quantifies the distributional organization of population activity. Population-state speed measures local deformation between successive population states.

\section{Neural HRSM state-space projection}

\subsection{Raw proxy definitions}

Each population state was first mapped into raw HRSM proxy coordinates. Let \(z(\cdot)\) denote robust median/MAD standardization, with a standard-deviation fallback when the median absolute deviation is numerically zero. For each time bin, the raw neural proxy coordinates were defined as:
\[
H_{\rm raw}
=
\frac{1}{4}
\left[
z(\bar r)
+
z(f_{\rm active})
+
z(\lVert r\rVert_2)
+
z(r_{\max})
\right],
\]
where \(\bar r\) is the population mean firing rate, \(f_{\rm active}\) is active-unit fraction, \(\lVert r\rVert_2\) is the population L2 rate norm, and \(r_{\max}\) is the maximum unit rate in the bin.

Recoverability-like return structure was defined as low local deformation:
\[
R_{\rm raw} = -z(v_{\rm state}),
\]
where \(v_{\rm state}\) is population-state speed.

The raw stability coordinate combined distributional organization with low local dispersion and low local deformation:
\[
S_{\rm raw}
=
\frac{1}{4}
\left[
z(E_r)
-
z(\sigma_r)
-
z(|\Delta \bar r|)
-
z(|\Delta E_r|)
\right],
\]
where \(E_r\) is population-rate entropy, \(\sigma_r\) is the across-unit rate standard deviation, and \(\Delta\) denotes the change from the previous bin within the same presentation trajectory.

The raw retained-history coordinate summarized available lagged population structure:
\[
M_{\rm raw}
=
w_{\rm hist}
\frac{1}{4}
\left[
z(\bar r_{t-1})
+
z(f_{{\rm active},t-1})
+
z(\lVert r_{t-1}\rVert_2)
+
z(E_{r,t-1})
\right],
\]
where \(w_{\rm hist}=1\) for bins with available lag history and \(w_{\rm hist}=0.25\) for the first bin of each trajectory after median lag imputation. This coordinate is a state-space retained-history proxy; the formal memory test is performed separately using aligned lag-history prediction and shuffled-lag controls.

The raw neural potential was:
\[
\Phi_{{\rm neural,raw}}
=
\frac{H_{\rm raw}+R_{\rm raw}+S_{\rm raw}+M_{\rm raw}}{4}.
\]

\subsection{Sequential residualization and axis audit}

The final \(H\), \(R\), \(S\), and \(M\) coordinates were obtained by sequential residualization in the order \(H \rightarrow R \rightarrow S \rightarrow M\). Thus \(H\) was retained as \(H_{\rm raw}\), \(R\) was residualized against \(H\), \(S\) was residualized against \(H\) and \(R\), and \(M\) was residualized against \(H\), \(R\), and \(S\). The final neural potential was:
\[
\Phi_{\rm neural}
=
\frac{H+R+S+M}{4}.
\]

The near-zero post-residualization correlations should not be interpreted as an independent biological discovery. They are a numerical verification that the residualization step worked. The biologically relevant audit is the raw pre-residualization correlation structure, which is reported alongside the post-residualization matrix.

\begin{table}[t]
\centering
\caption{Raw and orthogonalized HRSM axis audit. Raw correlations summarize pre-residualization axis overlap. Orthogonalized correlations verify numerical residualization and are not treated as independent biological evidence.}
\label{tab:axis-audit}
\input{paper/tables/table_strict_v2_axis_audit.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{results/figures/cross_session/allen_spontaneous_strict_v2/raw_axis_correlation_heatmap.png}
\caption{Mean raw HRSM axis correlation before residualization across the strict six-session cohort.}
\label{fig:raw-axis-corr}
\end{figure}

\section{Memory model and shuffled-lag control}

For each target \(y_t\), a baseline model was compared against a memory model containing aligned lag-history features. The main model used lags \((1,2)\). Additional ablation models used lag sets \((1)\), \((2)\), \((1,2)\), \((1,2,3)\), and \((1,2,3,4)\). Ridge regression was used in the main audits.

Raw memory gain was defined as:
\[
\Delta R^2_{\rm raw}
=
R^2_{\rm memory}
-
R^2_{\rm baseline}.
\]
To distinguish aligned historical contribution from distributional lag-feature effects, shuffled-lag controls were generated. The lag-feature rows were temporally shuffled relative to the target so that lag-feature distributions were retained while chronological alignment with the target was disrupted. Controlled memory gain was defined as:
\[
\Delta R^2_{\rm controlled}
=
\Delta R^2_{\rm observed}
-
\mathbb{E}
\left[
\Delta R^2_{\rm shuffled}
\right].
\]
This statistic is the primary memory quantity in the manuscript.

The shuffled-lag baseline is stricter than a simple uncontrolled lagged-model comparison because it preserves the marginal distribution of each lag component while disrupting its chronological alignment with the target. This control reduces the risk that apparent memory gain is driven only by target dispersion, static variance, or generic model inflation. A positive \(\Delta R^2_{\rm controlled}\) therefore indicates that the aligned recent past carries predictive structure beyond a temporally disrupted lag history.

\section{Results}

\subsection{Six-session strict synthesis}

The strict cohort produced a cross-session synthesis with six fixed-scale spontaneous sessions. Five of six sessions had LGd as the highest-\(\Phi_{\rm neural}\) region, while session 750749662 had CA1 as the highest-\(\Phi_{\rm neural}\) region. This prevents a universal LGd-dominance claim. Mean-rate memory was also heterogeneous: session 755434585 did not show all-region positive mean-rate controlled memory. These exceptions sharpen the interpretation: the robust result is not universal mean-rate memory, but target-level controlled organizational memory.

\begin{table}[t]
\centering
\caption{Strict six-session HRSM and memory summary. Session-level heterogeneity is retained rather than hidden.}
\label{tab:strict-session-summary}
\input{paper/tables/table_strict_v2_session_summary.tex}
\end{table}

\subsection{Controlled memory is strongest for recruitment and entropy}

Across six strict sessions, active-unit fraction and population-rate entropy were the strongest cross-session controlled-memory targets. The median observed-minus-shuffled gain was \StrictActiveMedianGain{} for active-unit fraction and \StrictEntropyMedianGain{} for population-rate entropy. Population-state speed ranked last.

\begin{table}[t]
\centering
\caption{Strict six-session target synthesis. Active-unit fraction and population-rate entropy are the strongest cross-session controlled-memory targets.}
\label{tab:strict-target-synthesis}
\input{paper/tables/table_strict_v2_target_synthesis.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{results/figures/cross_session/allen_spontaneous_strict_v2/cross_session_target_memory_heatmap.png}
\caption{Target-level controlled memory across the strict six-session cohort.}
\label{fig:strict-target-heatmap}
\end{figure}

\subsection{Variance and autocorrelation residualization preserves the leading organizational targets}

A variance-scaling audit tested whether controlled memory gain tracked target variance, dispersion, or lag-1 autocorrelation. In the strict six-session cohort, active-unit fraction and population-rate entropy remained the top two targets after residualizing target variance, lag-1 autocorrelation, and row count. This strengthens the organizational-memory interpretation relative to the earlier three-session audit.

\begin{table}[t]
\centering
\caption{Variance and autocorrelation residualization in the strict six-session cohort.}
\label{tab:strict-variance}
\input{paper/tables/table_strict_v2_variance_residual.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{results/figures/cross_session/allen_spontaneous_strict_v2/cross_session_variance_residual_target_rank.png}
\caption{Target ranking before and after variance/autocorrelation residualization.}
\label{fig:strict-variance}
\end{figure}

\subsection{Lag ablation supports a short distributed history window}

The six-session lag-ablation audit tested whether the memory contribution was exhausted by the nearest previous bin. It was not. Active-unit fraction and population-rate entropy remained positive across every tested lag set. Active-unit fraction increased from \StrictActiveLagOne{} under lag1 to \StrictActiveLagFour{} under lag1234. Population-rate entropy increased from \StrictEntropyLagOne{} under lag1 to \StrictEntropyLagFour{} under lag1234.

\begin{table}[t]
\centering
\caption{Strict six-session lag-ablation controlled gains. Values are median observed-minus-shuffled gains across sessions.}
\label{tab:strict-lag-ablation}
\input{paper/tables/table_strict_v2_lag_ablation.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{results/figures/ablation/allen_spontaneous_strict_v2/lag_ablation_controlled_memory_curves.png}
\caption{Lag-ablation curves for the strict six-session cohort. Active-unit fraction and population-rate entropy remain positive across lag settings and increase across a short recent-history window.}
\label{fig:strict-lag-ablation}
\end{figure}

This supports an operational interpretation in which spontaneous population history is distributed across a short recent temporal window rather than reducible to the immediately preceding state. The result remains bounded: it establishes a predictive historical contribution under this model class, not a complete mechanistic theory of neural memory.

\subsection{Ripeness remains descriptive}

A descriptive spontaneous neural ripeness index was computed from rank-normalized \(\Phi_{\rm neural}\), stability \(S\), controlled mean-rate memory, active-unit memory, and entropy memory. This index is explicitly post hoc and descriptive. It includes memory gains as inputs and therefore cannot be used as independent evidence that high-ripeness states have high memory. Its value is organizational: it separates high-\(\Phi_{\rm neural}\) structural states from states carrying strong controlled organizational memory.

\begin{table}[t]
\centering
\caption{Region-level descriptive ripeness summary for the strict six-session cohort.}
\label{tab:strict-ripeness}
\input{paper/tables/table_strict_v2_ripeness_region.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.78\linewidth]{results/figures/cross_session/allen_spontaneous_strict_v2/spontaneous_neural_ripeness_phi_vs_memory.png}
\caption{Structural potential versus organizational memory in the strict six-session cohort.}
\label{fig:strict-ripeness}
\end{figure}

\section{Discussion}

This analysis supports a bounded conclusion: spontaneous neural population activity contains controlled organizational memory. Across six strict fixed-scale Allen Neuropixels sessions, active-unit fraction and population-rate entropy are the strongest cross-session controlled-memory targets. The effect survives shuffled-lag controls, variance/autocorrelation residualization, and lag-ablation testing.

The result should not be interpreted as universal mean-rate memory. Session 755434585 is a useful stress test because mean-rate memory is not uniformly positive across regions, yet entropy and active-unit fraction remain the leading organizational targets. The result should also not be interpreted as independent external replication. All sessions come from the same Allen dataset, so the correct phrase is within-dataset cross-session generalization.

The target-level result is the conceptual center of the study. Active-unit fraction and population-rate entropy rank above mean firing rate in the strict cross-session synthesis. This suggests that retained history is most visible in how the population organizes itself: which units participate and how activity is distributed across the population. The lag-ablation audit reinforces this interpretation by showing that recruitment and entropy remain positive across all tested lag windows and increase as additional recent lags are included.

The HRSM coordinate audit also constrains the interpretation. Raw HRSM proxy axes are not assumed to be independent. Sequential residualization creates orthogonalized coordinates for state-space projection and downstream summaries. Therefore, near-zero post-residualization correlations verify the computation rather than proving independent biological axes. The raw-axis audit is included to make this distinction explicit.

Population-state speed remains the weakest target. This weak support is informative. State speed is a derivative-like quantity describing how rapidly the population state moves, whereas the present memory model uses recent population-position features. The result is therefore consistent with the interpretation that the retained-history signal is primarily structural and organizational, not kinetic. A stronger test of velocity or acceleration memory would require models that explicitly include velocity-like or acceleration-like predictors.

\section{Limitations}

The analysis remains within one public dataset. The six-session expansion strengthens within-dataset generalization but does not provide independent external replication. The HRSM coordinates are operational proxies and should not be treated as direct physiological variables. The residualized axes are useful for reduced state-space analysis, but their orthogonality is produced by the residualization procedure. The ripeness index is descriptive and partly circular because memory gains contribute to the score.

The shuffled-lag control is strong but not exhaustive. Other controls, including blockwise temporal shifts, circular shifts, region-label permutations, alternative model classes, and held-out unit resampling, should be tested in future work. The extraction scale is fixed at twenty units per region in the strict cohort; future sensitivity tests should evaluate 10-, 20-, and 30-unit extraction scales where data coverage allows.

\section{Conclusion}

A reduced HRSM state-space analysis of six strict fixed-scale Allen Neuropixels spontaneous sessions reveals within-dataset cross-session generalization of controlled organizational memory. Active-unit fraction and population-rate entropy are the strongest cross-session targets. Both survive variance/autocorrelation residualization and remain positive across lag-ablation settings. Mean-rate memory is present but heterogeneous, and state-speed memory remains weak. The resulting claim is deliberately bounded: spontaneous neural population activity carries controlled organizational memory across a short recent history window.

\section*{Data availability}

No new animal data were generated. The analysis uses publicly available Allen Visual Coding Neuropixels data. Derived summaries, figures, and scripts are stored in the accompanying repository under the spontaneous Allen HRSM branch.

\section*{Code availability}

The analysis scripts are stored under \texttt{scripts/allen/} and \texttt{scripts/paper/}. The strict-v2 outputs used in this manuscript are stored under \texttt{results/cross\_session/allen\_spontaneous\_strict\_v2/}, \texttt{results/ablation/allen\_spontaneous\_strict\_v2/}, and \texttt{results/figures/}. A frozen reproducibility tag should be generated before submission.

\section*{Ethics statement}

This study reuses publicly released Allen Brain Observatory data. No new animal experiments were performed.

\section*{Competing interests}

The author declares no competing interests.

\section*{Author contributions}

S.A. designed the HRSM analysis framework, implemented the computational workflow, analyzed the Allen Neuropixels data, generated figures and robustness audits, and wrote the manuscript.

\section*{Acknowledgements}

The author acknowledges the Allen Institute for publicly releasing the Visual Coding Neuropixels dataset and associated documentation.

\bibliographystyle{plainnat}
\bibliography{paper/references}

\appendix

\section{Strict cohort and diagnostic session status}

The strict cohort contains six sessions with four regions and twenty units per region. Session 754312389 is excluded from the strict cohort and retained as a partial-coverage diagnostic because one region contributed fewer than twenty units.

\section{Machine-readable robustness flags}

The main machine-readable flags are stored in:
\[
\texttt{results/cross\_session/allen\_spontaneous\_strict\_v2/}
\]
and
\[
\texttt{results/ablation/allen\_spontaneous\_strict\_v2/}.
\]
The most important passing flags are that active-unit fraction and population-rate entropy are the top two raw and residualized cross-session targets, and that both remain positive across all lag-ablation settings. The most important failing flags are that universal all-region mean-rate memory and universal LGd dominance are not supported.

\end{document}
"""


def main():
    out = Path("paper/neural_hrsm_spontaneous_memory_strict_v2.tex")
    out.write_text(TEX)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
