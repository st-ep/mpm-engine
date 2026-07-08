# Claymore design notes (for warpmpm Step 5)

Source: the local checkout at ../claymore = Justin Bonus's fork (ClaymoreUW lineage,
github.com/JustinBonus/claymore, master 022ae31) of penn-graphics-research/claymore,
Wang, Qiu et al., "A Massively Parallel and Scalable Multi-GPU Material Point Method",
ACM TOG 39(4) (SIGGRAPH 2020), doi 10.1145/3386569.3392442. LICENSE: MIT (verified in
the checkout; the earlier GPLv3 caution in speedup_plan.md applied to the 2018 GPUMPM
repo, NOT to claymore). Bonus's fork adds the OSU_LWF wave-flume and debris projects,
multi-arch builds, and engineering-focused materials/output; the transfer kernels keep
the upstream architecture. We may read and port from this code with citation
(AUTHORS.md has the entry).

Reference files read: Projects/OSU_LWF/{settings.cuh, mgmpm_kernels.cuh,
particle_buffer.cuh}, Library/MnSystem grid/partition structures.

## Their architecture, reduced to the five load-bearing ideas

1. FUSED G2P2G. One kernel does: gather grid(n) -> update particle (F, stress inside
   the same kernel) -> scatter to grid(n+1). Grid is double-buffered (grid, next_grid);
   particle buffers double-buffered likewise. Each substep is g2p2g + grid_update only.
   Particle arrays are read ONCE and written ONCE per substep instead of 3x (stress,
   p2g, g2p passes), and intermediate particle state never round-trips global memory.

2. BLOCK-CENTRIC LAUNCH + SHARED-MEMORY ARENAS. One CUDA block per active 4^3 grid
   block. Cooperative load of the (2*4)^3 arena (block + halo) of grid velocities to
   shared memory; per-particle work reads/writes shared; block-local atomics; one
   cooperative writeback with global atomics only at arena edges. This is what removes
   global-memory atomic contention in P2G.

3. PARTICLE BUCKETS PER BLOCK. partition._blockbuckets/_ppbs/_binsts: particles are
   binned to their stencil-base block every step; the g2p2g block loops its own bucket.
   Rebinned by a cheap counting pass, not a full sort ("advection buckets" carry the
   particle's block delta).

4. AoSoA PARTICLE BINS. Particles stored in bins of g_bin_capacity (32/64) with
   channels laid out per bin (x0..x31, y0..y31, ...). A warp reading channel c of its
   bin is fully coalesced. Bins belong to blocks, so bucket order == memory order.

5. MATERIAL-TEMPLATED KERNELS. One compiled g2p2g per material keeps register
   pressure bounded (their per-material template specializations).

## What ports to warpmpm tonight (CPU-testable), and what does not

- (1) PORTS as a particle-centric fused kernel WITHOUT shared arenas: warp kernels
  cannot express cooperative shared-memory arenas (that needs wp.tile or native CUDA
  snippets, GPU-only, unverifiable on this Mac). But our grid already has the double
  buffer built in: p2g scatters to grid_v_in/grid_m, normalize writes grid_v_out, BC
  and g2p consume grid_v_out. So a fused kernel can read grid_v_out (state n) and
  scatter into the zeroed grid_v_in/grid_m (state n+1) with no aliasing.
  Pipeline per tick (S substeps):
    substep 0:            zero_all -> stress -> p2g            (prologue)
    substeps 1..S-1:      zero_m_vin -> FUSED(g2p+stress+p2g) -> zero_vout
    every substep:        normalize -> BC
    after last BC:        g2p                                   (epilogue)
  = S+1 particle passes per S substeps instead of 3S.
  The split zero is load-bearing for bitwise equality: normalize writes v_out only
  where mass > 1e-15, so sub-threshold stencil nodes must see a ZEROED v_out, but the
  fused kernel must read the pre-zero v_out of state n. Hence zero m+v_in before the
  fused pass and v_out after it.
- (3) PORTS as a periodic block-key sort (host argsort at the guard readback, device
  permutation through scratch, wp.copy back in place to keep array pointers stable for
  the CUDA-graph contract). Claymore rebins every step because shared-memory g2p2g
  REQUIRES bucket order; our global-atomic kernels only want locality, so every K
  ticks is enough.
- (4) PARTIALLY COVERED by (3): after a block sort, our SoA arrays are block-contiguous,
  which restores the coalescing AoSoA provides within a warp. The remaining AoSoA
  delta (channel interleaving per bin, fewer TLB streams) only matters with the
  shared-memory kernel of (2), and is unmeasurable on CPU. Verdict: implement (1)+(3),
  measure on GPU, take (2)+(4) together as a CUDA-native follow-up if the fused
  pipeline is not enough.
- (5) PARTIALLY EXISTS: our kernels branch on material id per particle; warp compiles
  one fat kernel. If GPU register pressure hurts the fused kernel, split it per
  material family (fluid-like vs SVD-needing) the way claymore templates do.
