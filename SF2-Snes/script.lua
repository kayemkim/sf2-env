previous_p1_health = 176
previous_p2_health = 176

function check_reward()
  p1_health_delta = previous_p1_health - data.p1_health
  p2_health_delta = previous_p2_health - data.p2_health
  
  previous_p1_health = data.p1_health
  previous_p2_health = data.p2_health
    
  return p2_health_delta - p1_health_delta
end

function check_reward_with_time_penalty()
  reward = check_reward_by_health()
  if reward == 0 then
    return reward - 1
  end
  return reward
end

function done_check()
  return data.p1_health == 255 or data.p2_health == 255 or data.timer <= 0
end

function done_check_sunppang()
  return data.p1_health ~= data.p2_health
end


