package com.deepnetts.examples.regression.bostonhousing.pipeline;

import com.deepnetts.examples.regression.bostonhousing.dto.Neighborhood;

import java.util.List;

// Turns a neighborhood into the numbers the network uses.
public final class FeatureExtractor {

    // Name and order of every input. The order must match extract().
    public static final List<String> FEATURE_NAMES = List.of(
            "log_crime_rate", "avg_rooms", "old_buildings_percent", "distance_to_employment_km",
            "log_highway_access_index", "pupil_teacher_ratio", "nitrogen_oxides_ppm", "lower_status_percent");

    public static final int NUM_FEATURES = FEATURE_NAMES.size();

    // The largest values in the data. Dividing by them puts every input roughly between 0 and 1,
    // so no input dominates.
    private static final double MAX_CRIME_RATE        = 90;
    private static final double MAX_ROOMS             = 9;
    private static final double MAX_PERCENT           = 100;
    private static final double MAX_DISTANCE_KM       = 13;
    private static final double MAX_HIGHWAY_INDEX     = 24;
    private static final double MAX_PUPIL_TEACHER     = 22;
    private static final double MAX_NITROGEN_OXIDES   = 0.9;
    private static final double MAX_LOWER_STATUS      = 38;

    private FeatureExtractor() {
    }

    public static float[] extract(Neighborhood n) {
        return new float[] {
                // log1p(x) is log(1 + x). Most neighborhoods have little crime, a few have a lot:
                // using log so the extreme ones don't dominate. The same goes for highway access.
                (float) (Math.log1p(n.crimeRate()) / Math.log1p(MAX_CRIME_RATE)),
                (float) (n.avgRooms() / MAX_ROOMS),
                (float) (n.oldBuildingsPercent() / MAX_PERCENT),
                (float) (n.distanceToEmploymentKm() / MAX_DISTANCE_KM),
                (float) (Math.log1p(n.highwayAccessIndex()) / Math.log1p(MAX_HIGHWAY_INDEX)),
                (float) (n.pupilTeacherRatio() / MAX_PUPIL_TEACHER),
                (float) (n.nitrogenOxidesPpm() / MAX_NITROGEN_OXIDES),
                (float) (n.lowerStatusPercent() / MAX_LOWER_STATUS)
        };
    }
}
