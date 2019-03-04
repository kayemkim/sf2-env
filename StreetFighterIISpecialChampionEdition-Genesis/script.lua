previous_agent_health = 176
previous_cpu_health = 176

function check_reward_by_health()
  agent_health_delta = previous_agent_health - data.first_player_health
  cpu_health_delta = previous_cpu_health - data.second_player_health
  
  previous_agent_health = data.first_player_health
  previous_cpu_health = data.second_player_health
    
  return cpu_health_delta - agent_health_delta - 1
end

function done_check()
  if data.continuetimer == 0 then
    return true
  end
  if data.first_player_matches_won == 1 then
    return true
  end
  if data.second_player_matches_won == 1 then
    return true
  end
  return false
end
