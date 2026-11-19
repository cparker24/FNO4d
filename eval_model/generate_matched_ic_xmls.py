#!/usr/bin/env python
"""Generate the 240 (4 centralities x 20 events x 3 arms) matched-initial-
condition particlization XML configs, following the structure of the
existing single-centrality/5-event precedent
(config/particlization_investigation/fno_3d_particlization_10_20_fno_matched_ic_eta4_n65_event{0..4}_noviscous.xml
and its ..._ground_truth_... sibling), generalized to:
  - all 4 centralities (that precedent only covered 10-20%)
  - the full 20-event validation set (that precedent only covered 5 events)
  - the symmetric-grid, 4-channel (notemp, drop_channels:[1]) old and
    doubled-data checkpoints (that precedent used the older 5-channel,
    cropped-grid "widened_eta4_n65" checkpoint) -- so x_min/y_min/eta_min
    are -9.84375/-9.84375/-3.9375 (symgrid) not -10.0/-10.0/-4.0, and
    n_features is 4 not 5, matching this session's
    fno_3d_particlization_{cen}_fno_eta4_n65_symgrid_doubled_noviscous_20ev.xml
    siblings.
  - a third arm (old vs. doubled vs. MUSIC ground-truth-passthrough), not
    just old-FNO vs. ground-truth.

Model files are expected at
FnoHydro/models/matched_ic/{fixed_ic_old,fixed_ic_doubled,ground_truth_passthrough}_symgrid_{cen}_event{N}.pt
(built by build_matched_ic_models_batch.py).
"""
from pathlib import Path

CONFIG_DIR = Path(
    "/home/cparker/jetscape-docker/X-SCAPE/external_packages/js-contrib/contribs/FnoHydro/"
    "config/particlization_investigation"
)
CENTRALITIES = {
    "0_10": (0, 10),
    "10_20": (10, 20),
    "20_30": (20, 30),
    "30_40": (30, 40),
}
N_EVENTS = 20
ARMS = {
    "old": "fixed_ic_old_symgrid_{cen}_event{event}.pt",
    "doubled": "fixed_ic_doubled_symgrid_{cen}_event{event}.pt",
    "music": "ground_truth_passthrough_symgrid_{cen}_event{event}.pt",
}

TEMPLATE = """<?xml version="1.0" ?>
<jetscape>

  <!--
    Matched-initial-condition particlization comparison (old-FNO vs.
    doubled-data FNO vs. MUSIC ground truth), generalized from the 10-20%/
    5-event precedent (see PARTICLIZATION_INVESTIGATION.md's "Redone with a
    matched initial condition" section) to the full 20-event validation set
    across all 4 centralities, on the symmetric-grid/4-channel (notemp)
    checkpoints this session's doubled-data benchmark uses.

    Arm: {arm}, centrality {cen_low}-{cen_high}%, event {event}
    (AuAu_valid20ev_{cen}_eta4_n65_symgrid.root).

    <model_file> is a "{arm_kind}" TorchScript wrapper
    (FNO4d/eval_model/build_matched_ic_models_batch.py) that ignores its
    (live-primed, irrelevant) input and always returns this one validation
    event's own data -- {arm_desc}.

    Known caveat (not fixed, already quantified as negligible in the
    precedent): FnoHydro.cc always prepends the live MCGlauber+MusicPrimedPreEq
    snapshot as the trajectory's position-0 slice regardless of what the
    model returns, so that single slice (1/120) is still a live, physically
    unrelated event; every other slice (119/120) is genuinely this event's
    own data.

    Siblings: the other two arms for the same (centrality, event); the
    unmatched-IC 20-event ensemble comparison earlier this session
    (particlization_{cen}_fno_eta4_n65_symgrid_doubled_noviscous_20ev.xml and
    siblings).
  -->

  <enableAutomaticTaskListDetermination
    > false </enableAutomaticTaskListDetermination>

  <nEvents> 1 </nEvents>

  <Random>
    <seed>424242</seed>
  </Random>

  <vlevel> 0 </vlevel>

  <setReuseHydro> false </setReuseHydro>
  <nReuseHydro> 0 </nReuseHydro>

  <outputFilename>matched_ic_{cen}_{arm}_event{event}_noviscous.dat</outputFilename>
  <JetScapeWriterAscii> on </JetScapeWriterAscii>

  <write_pthat> 1 </write_pthat>

  <!-- Hard Process (required by fnoHydroTest.cc's hardcoded PGun) -->
  <Hard>
    <PGun>
        <name>PGun</name>
        <pT>20</pT>
        <parID>21</parID>
    </PGun>
  </Hard>

  <!-- Inital State Module: MCGlauber. Irrelevant to the fixed-IC/ground-truth
       wrapper's output (see header), kept identical to every sibling config
       regardless -- only used to drive JETSCAPE's own event loop and
       MusicPrimedPreEq's live priming (whose output is discarded except for
       the single position-0 seam slice, see header). -->
  <IS>
      <grid_max_x> 10 </grid_max_x>
      <grid_max_y> 10 </grid_max_y>
      <grid_max_z> 6 </grid_max_z>
      <grid_step_x> 0.3 </grid_step_x>
      <grid_step_y> 0.3 </grid_step_y>
      <grid_step_z> 0.2 </grid_step_z>

      <MCGlauber>
        <generateOnlyPositions>0</generateOnlyPositions>
        <projectile>Au</projectile>
        <target>Au</target>
        <sqrts>200</sqrts>
        <b_min>0.0</b_min>
        <b_max>20.0</b_max>
        <cenMin>{cen_low}.0</cenMin>
        <cenMax>{cen_high}.0</cenMax>
        <nucleon_configuration_from_file>0</nucleon_configuration_from_file>
        <useQuarks>1</useQuarks>
        <Q2>1.0</Q2>
        <shadowing_factor>0.0</shadowing_factor>
        <baryon_junctions>0</baryon_junctions>
        <N_sea_partons>1</N_sea_partons>
        <d_min>0.9</d_min>
        <lambdaB>0.2</lambdaB>
        <BG>3.960</BG>
        <Subtract_hard_momentum>0</Subtract_hard_momentum>
        <lambdaBs>0.0</lambdaBs>
        <baryonInStringProb>0.0</baryonInStringProb>
        <fluct_Nstrings_per_NN_collision>1</fluct_Nstrings_per_NN_collision>
        <QCD_string_production_mode>1</QCD_string_production_mode>
        <rapidity_loss_method>4</rapidity_loss_method>
        <ylossParam4At2>1.610</ylossParam4At2>
        <ylossParam4At4>1.685</ylossParam4At4>
        <ylossParam4At6>1.685</ylossParam4At6>
        <ylossParam4At10>2.95</ylossParam4At10>
        <ylossParam4var>0.682</ylossParam4var>
        <remnant_energy_loss_fraction>0.536</remnant_energy_loss_fraction>
        <fluctuation_remnant_energy_loss_fraction>0</fluctuation_remnant_energy_loss_fraction>
        <remnant_energy_loss_fraction_var>0.4</remnant_energy_loss_fraction_var>
        <evolve_QCD_string_mode>4</evolve_QCD_string_mode>
        <tau_form_mean>0.424</tau_form_mean>
        <tau_form_fluct_gamma_beta>1.0</tau_form_fluct_gamma_beta>
      </MCGlauber>
  </IS>

  <!--Preequilibrium Dynamics Module: MusicPrimedPreEq (see driver .cc),
       driven by <taus> below as its priming target tau. -->
  <Preequilibrium>
      <tau0>0.44</tau0>
      <taus>0.44</taus>
      <NullPreDynamics> </NullPreDynamics>
  </Preequilibrium>

  <!-- Hydro Module: live 3D+1 FnoHydro, with the matched-IC wrapper in
       place of the normal (live-primed-input) model. <MUSIC> here is read
       by MusicPrimedPreEq's EMBEDDED MpiMusic instance (the "priming"
       stage, InitialProfile=131) -- not by the actual Hydro task for this
       run, which is FnoHydro (<FNO> below). -->
  <Hydro>
      <MUSIC>
          <name>MUSIC</name>
          <MUSIC_input_file>music_input</MUSIC_input_file>
          <EOS>91</EOS>
          <InitialProfile>131</InitialProfile>
          <beastMode>1</beastMode>
          <Initial_time_tau_0>0.424</Initial_time_tau_0>
          <output_evolution_to_file>0</output_evolution_to_file>
          <output_evolution_to_memory>1</output_evolution_to_memory>
          <output_evolution_every_N_timesteps>1</output_evolution_every_N_timesteps>
          <surface_in_memory>1</surface_in_memory>

          <shear_viscosity_eta_over_s>0.108</shear_viscosity_eta_over_s>
          <T_dependent_Shear_to_S_ratio>3</T_dependent_Shear_to_S_ratio>
          <shear_viscosity_3_T_kink_in_GeV>0.206</shear_viscosity_3_T_kink_in_GeV>
          <shear_viscosity_3_at_kink>0.108</shear_viscosity_3_at_kink>
          <shear_viscosity_3_low_T_slope_in_GeV>-1.999</shear_viscosity_3_low_T_slope_in_GeV>
          <shear_viscosity_3_high_T_slope_in_GeV>1.999</shear_viscosity_3_high_T_slope_in_GeV>
          <muB_dependent_Shear_to_S_ratio>0</muB_dependent_Shear_to_S_ratio>

          <temperature_dependent_bulk_viscosity>3</temperature_dependent_bulk_viscosity>
          <bulk_viscosity_3_max>0.092</bulk_viscosity_3_max>
          <bulk_viscosity_3_T_peak_in_GeV>0.180</bulk_viscosity_3_T_peak_in_GeV>
          <bulk_viscosity_3_width_in_GeV>0.067</bulk_viscosity_3_width_in_GeV>
          <bulk_viscosity_3_lambda_asymm>-0.8</bulk_viscosity_3_lambda_asymm>

          <Include_second_order_terms>1</Include_second_order_terms>
          <Include_Rhob>0</Include_Rhob>
          <Include_QS>0</Include_QS>
          <use_eps_for_freeze_out>0</use_eps_for_freeze_out>
          <Do_FreezeOut_lowtemp>1</Do_FreezeOut_lowtemp>
          <average_surface_over_this_many_time_steps>5</average_surface_over_this_many_time_steps>
          <Perform_CooperFrye_Freezeout>0</Perform_CooperFrye_Freezeout>
          <string_source_sigma_x>0.207</string_source_sigma_x>
          <string_source_sigma_eta>0.458</string_source_sigma_eta>
          <stringPreEqFlowFactor>0.0</stringPreEqFlowFactor>

          <freezeout_temperature>0.150</freezeout_temperature>
      </MUSIC>

      <FNO>
        <model_file
            >../external_packages/js-contrib/contribs/FnoHydro/models/matched_ic/{model_file}</model_file>

        <centrality-low>{cen_low}</centrality-low>
        <centrality-high>{cen_high}</centrality-high>
        <EOS_id_MUSIC>91</EOS_id_MUSIC>

        <n_features>4</n_features>
        <energy_density_normalization>etau75_3d</energy_density_normalization>
        <max_rollout_windows>1</max_rollout_windows>
        <freezeout_temperature>0.150</freezeout_temperature>

        <nx>64</nx>
        <ny>64</ny>
        <neta>64</neta>
        <ntau>119</ntau>

        <x_min>-9.84375</x_min>
        <y_min>-9.84375</y_min>
        <eta_min>-3.9375</eta_min>

        <dtau>0.1</dtau>
        <deta>0.125</deta>
      </FNO>
  </Hydro>

  <!-- Surface sampler: must match every sibling config's oversampling and
       iSS viscous-correction settings exactly. -->
  <SoftParticlization>
    <iSS>
      <number_of_repeated_sampling>20</number_of_repeated_sampling>
      <include_deltaf_shear>0</include_deltaf_shear>
      <include_deltaf_bulk>0</include_deltaf_bulk>
    </iSS>
  </SoftParticlization>

  <!--Eloss Modules -->
    <Eloss>
        <maxT>100</maxT>
      <Matter>
        <Q0> 2.0 </Q0>
        <in_vac> 0 </in_vac>
        <vir_factor> 0.25 </vir_factor>
        <recoil_on> 1 </recoil_on>
        <broadening_on> 0 </broadening_on>
        <hydro_Tc> 0.16 </hydro_Tc>
      </Matter>
    </Eloss>

    <!-- Jet Hadronization Module -->
    <JetHadronization>
      <name>colorless</name>
    </JetHadronization>

</jetscape>
"""

ARM_KIND = {
    "old": "fixed-IC FNO (pre-doubling checkpoint)",
    "doubled": "fixed-IC FNO (doubled-training-data checkpoint)",
    "music": "ground-truth passthrough",
}
ARM_DESC = {
    "old": "the real trained pre-doubling network, run on this event's own tau0 state",
    "doubled": "the real trained doubled-data network, run on this event's own tau0 state",
    "music": "real MUSIC evolution for this event, downsampled onto the FNO grid -- not a model prediction at all",
}


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for cen, (cen_low, cen_high) in CENTRALITIES.items():
        for event in range(N_EVENTS):
            for arm, model_tmpl in ARMS.items():
                model_file = model_tmpl.format(cen=cen, event=event)
                xml = TEMPLATE.format(
                    arm=arm, cen=cen, cen_low=cen_low, cen_high=cen_high, event=event,
                    model_file=model_file, arm_kind=ARM_KIND[arm], arm_desc=ARM_DESC[arm],
                )
                out_path = CONFIG_DIR / f"fno_3d_particlization_matched_ic_{cen}_{arm}_event{event}_noviscous.xml"
                out_path.write_text(xml)
                count += 1
    print(f"Wrote {count} XML configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
