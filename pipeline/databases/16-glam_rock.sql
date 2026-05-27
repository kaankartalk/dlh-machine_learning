-- lists glam rock bands by longevity
SELECT band_name,
       (CASE
            WHEN split = '?' OR split IS NULL THEN 2020 - formed
            ELSE split - formed
        END) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
