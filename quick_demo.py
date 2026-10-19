"""Quick demo (about 1 minute): evaluate the pretrained VIGIL-LoRa policy against ADR and VoI-threshold."""
import evaluate as ev
print(f"{'Method':<15}{'NMSE':>8}{'Outage':>9}{'PDR':>7}{'Delay(min)':>12}{'TxEnergy(J/day)':>17}")
for name, pol, sh in [('ADR', ev.pol_adr(), False), ('VoI-threshold', ev.pol_voi(), False),
                      ('VIGIL-LoRa', ev.pol_net('models/vigil.pkl'), True)]:
    r = ev.run(pol, N=200, seed=0, days=1, shield=sh)
    print(f"{name:<15}{r['nmse']:>8.3f}{r['outage']:>9.3f}{r['pdr']:>7.3f}{r['delay_min']:>12.2f}{r['energy_J_node_day']:>17.2f}")
