Trial {trial_idx} finished.
Score: {score}  (best so far: {best_score})
Success: {success}

Score breakdown:
{breakdown}

Positions (mm, robot frame):
  Object start: {object_start}
  Object end:   {object_end}
  Target:       {target}

Aborted reason: {aborted_reason}
Safety violations: {safety_reasons}

Current PARAMS:
{current_params}

Best-so-far PARAMS:
{best_params}

Attached images: trial start frame, trial end frame.

Propose your next PARAMS change as a JSON object with only the keys you want
to update. Respond with JSON only.