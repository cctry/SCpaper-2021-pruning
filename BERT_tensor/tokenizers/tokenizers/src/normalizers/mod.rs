pub mod bert;
pub mod strip;
pub mod unicode;
pub mod utils;

pub use crate::normalizers::bert::BertNormalizer;
pub use crate::normalizers::strip::Strip;
pub use crate::normalizers::unicode::{NFC, NFD, NFKC, NFKD};
pub use crate::normalizers::utils::{Lowercase, Sequence};

use serde::{Deserialize, Serialize};

use crate::{NormalizedString, Normalizer};

/// Wrapper for known Normalizers.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NormalizerWrapper {
    BertNormalizer(BertNormalizer),
    StripNormalizer(Strip),
    NFC(NFC),
    NFD(NFD),
    NFKC(NFKC),
    NFKD(NFKD),
    Sequence(Sequence),
    Lowercase(Lowercase),
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> crate::Result<()> {
        match self {
            NormalizerWrapper::BertNormalizer(bn) => bn.normalize(normalized),
            NormalizerWrapper::StripNormalizer(sn) => sn.normalize(normalized),
            NormalizerWrapper::NFC(nfc) => nfc.normalize(normalized),
            NormalizerWrapper::NFD(nfd) => nfd.normalize(normalized),
            NormalizerWrapper::NFKC(nfkc) => nfkc.normalize(normalized),
            NormalizerWrapper::NFKD(nfkd) => nfkd.normalize(normalized),
            NormalizerWrapper::Sequence(sequence) => sequence.normalize(normalized),
            NormalizerWrapper::Lowercase(lc) => lc.normalize(normalized),
        }
    }
}

impl_enum_from!(BertNormalizer, NormalizerWrapper, BertNormalizer);
impl_enum_from!(NFKD, NormalizerWrapper, NFKD);
impl_enum_from!(NFKC, NormalizerWrapper, NFKC);
impl_enum_from!(NFC, NormalizerWrapper, NFC);
impl_enum_from!(NFD, NormalizerWrapper, NFD);
impl_enum_from!(Strip, NormalizerWrapper, StripNormalizer);
impl_enum_from!(Sequence, NormalizerWrapper, Sequence);
impl_enum_from!(Lowercase, NormalizerWrapper, Lowercase);
