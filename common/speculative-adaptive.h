#pragma once

#include <algorithm>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Single-bucket state machine (from the stew675 delivery, `bucketed-port/tuned-port.patch`, itself the
// recommended responsive form of upstream PR #27210's stricter table). Every verification result moves a
// bucket B by (n_accepted - depth), except that a full accept (n_accepted >= depth) is credited
// max(1, n_accepted - 1) instead, so a longer accept is worth more and a strong run climbs faster.
// A miss of k tokens subtracts k.
//
// B starts at the drop pressure D on every depth change. When B reaches the cap T = D + climb budget the
// depth climbs one step; when B falls to 0 the depth drops one step. Surplus above the cap (bounded by the
// depth, so at most one step per update) carries into the new depth, and a deficit below zero carries down
// the same way.
//
// Constants (delivery defaults): D = max(60, 10*depth); climb budget = 20 + 6*(depth-1).
// The upstream PR #27210 table required climb_threshold(3) = 10 CONSECUTIVE FULL accepts to leave the
// floor, which at our measured ~50% acceptance almost never happens - the controller sat pinned at its
// floor and behaved exactly like fixed n3. The bucket form needs only NET surplus, so it engages.
struct common_speculative_adaptive {
    int n_cur    = 0; // current adaptive draft depth N
    int n_bucket = 0; // accumulated net accepted surplus over the depth

    // accumulated (depth - accepted) needed to drop one step from the current depth;
    // flat deep floor at 60, then 10 per depth so deep depths still fall in a few misses
    int drop_pressure() const {
        return std::max(60, 10 * n_cur);
    }

    // climb budget: 20, growing 6 per depth so climbing high has to be earned
    int climb_budget() const {
        return 20 + 6 * (n_cur - 1);
    }

    // bucket cap: drop pressure + climb budget, so from the reset point a climb costs the budget
    int bucket_cap() const {
        return drop_pressure() + climb_budget();
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max.
    // Cold start begins `cap - 3` deep when that is above the floor: the drift then pulls the depth
    // DOWN (cheap, even for a reasoning workload) instead of having to earn its way up from the floor.
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur    = std::min(cap, std::max(floor, cap - 3));
        n_bucket = drop_pressure();
    }

    // feed one verification result: n_accepted is the number of tokens the target accepted this round
    // (it can exceed the current depth when another speculator produced the accepted draft)
    void update(int n_accepted, int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        const int delta = (n_accepted >= n_cur) ? std::max(1, n_accepted - 1) : (n_accepted - n_cur);

        n_bucket += delta;

        if (n_cur < cap && n_bucket >= bucket_cap()) {
            // climb one step; carry the surplus above the cap, bounded by the depth so a lucky streak
            // cannot cascade up several depths in one update
            const int carry = std::min(n_bucket - bucket_cap(), n_cur);
            n_cur++;
            n_bucket = drop_pressure() + carry;
        } else if (n_cur > floor && n_bucket <= 0) {
            // drop one step; carry the deficit
            const int carry = n_bucket;
            n_cur--;
            n_bucket = drop_pressure() + carry;
        } else if (n_cur <= floor) {
            n_bucket = std::max(n_bucket, 0); // at the floor, clamp instead of going negative
        }
    }
};
