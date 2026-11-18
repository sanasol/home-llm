# Attribution and reproduction limits

The SPEC1-w15 table in `spec1_w15.txt`, tensor-core tile permutation, trellis
storage conventions and public decoder equations in `stq.py` derive from the
public [beamivalice/sushi](https://github.com/beamivalice/sushi) reference at
`d5e876edcb1538a347041b1b7cc28a0e6e8f0af6`. Its unmodified license is retained in
`LICENSE-SUSHI`. Recipe orchestration and the F32 search implementation are
published under the folder's MIT `LICENSE`.

Credit to DeepSeek for the native model, beamster/beamivalice for the official
Sushi pack and consumer format, and pqhaz for the Heretic FP8 patch. The weights
are downloaded from their separately licensed upstream repositories, not bundled
here. The patch's original F8_E4M3 tensor bytes are decoded with the pinned native
base scales; the recipe changes only 38 attention output trellis banks in an
otherwise official Sushi root.

This is a cleaned public version of the historical overlay recipe. Explicit
revision pins and portable paths were added, unrelated donor modes were removed,
and an independent root assembler/check was supplied. It does not guarantee byte
identity with historical GPU outputs across library/device versions and does not
reproduce the private Sashimi calibrated encoder. Never characterize it as a full
Heretic checkpoint requantization or a full quality evaluation.
