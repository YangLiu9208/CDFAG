% response to have 0 mean
function z_m_y = zero_mean_y(y),
    z_m_y = y - mean(y);
end