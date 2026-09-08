from wiemip_registry.BEPS.convert import BEPS
from wiemip_registry.BiomeE.convert import BiomeE
from wiemip_registry.CLASSIC.convert import CLASSIC
from wiemip_registry.CLM.convert import CLM
from wiemip_registry.CLM_FATES.convert import CLM_FATES
from wiemip_registry.DLEM.convert import DLEM
from wiemip_registry.JSBACH.convert import JSBACH
from wiemip_registry.JULES.convert import JULES
from wiemip_registry.LPJ_EOSIM.convert import LPJ_EOSIM
from wiemip_registry.LPJ_GUESS.convert import LPJ_GUESS
from wiemip_registry.LPJmL6.convert import LPJmL6
from wiemip_registry.LPX_Bern.convert import LPX_Bern
from wiemip_registry.TEM.convert import TEM
from wiemip_registry.VISIT_UT.convert import VISIT_UT
from wiemip_registry.core import Model

adapters = {
    "BEPS": BEPS(),
    # "BiomeE": BiomeE(),
    "CLASSIC": CLASSIC(),
    "CLM": CLM(),
    "CLM_FATES": CLM_FATES(),
    "DLEM": DLEM(),
    "JSBACH": JSBACH(),
    "JULES": JULES(),
    # "LPJ_EOSIM": LPJ_EOSIM(),
    "LPJ_GUESS": LPJ_GUESS(),
    "LPJmL6": LPJmL6(),
    "LPX_Bern": LPX_Bern(),
    "VISIT_UT": VISIT_UT(),
    "TEM": TEM(),
}

models = tuple(Model(name, adapter) for name, adapter in adapters.items())
