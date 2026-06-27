function stop = email_bayesopt_outputfcn(BO, State, email_addr, run_label)
% email_bayesopt_outputfcn  Implement email bayesopt outputfcn for this MATLAB workflow.
% Inputs: BO, State, email_addr, run_label
% Outputs: stop
%EMAIL_BAYESOPT_OUTPUTFCN  Send an email after each completed bayesopt iter.
%
%   stop = email_bayesopt_outputfcn(BO, State, email_addr, run_label)
%
%   Designed to be added to bayesopt's OutputFcn cell, alongside @saveToFile
%   and any other fcns. Fires once per iteration (in parallel mode, that's
%   once per completed parallel batch's last finished trial). Body
%   includes: iter number, objective at this iter, best-so-far,
%   1-objective converted to F1 (assumes objective = 1 - F1, the
%   convention used by 'val_min_f1*' BO objectives), the HPs of the
%   completed trial, and the best HPs so far.
%
%   Errors in the mail send are caught and printed; bayesopt is never
%   stopped by an email failure (`stop` always returns false).

    stop = false;
    if ~strcmp(State, 'iteration')
        return;
    end

    try
        n = size(BO.XTrace, 1);
        if n == 0
            return;
        end

        last_obj   = BO.ObjectiveTrace(end);
        best_obj   = BO.MinObjective;
        last_X     = BO.XTrace(end, :);
        best_X     = BO.XAtMinObjective;
        last_runtm = BO.ObjectiveEvaluationTimeTrace(end);

        % Subject line: short, with iter and best F1 so the inbox is
        % browsable without opening every message.
        subj = sprintf('[%s] BO iter %d  F1=%.4f  best=%.4f', ...
            run_label, n, 1 - last_obj, 1 - best_obj);

        % Body: full iter context.
        body = sprintf([ ...
            'Run         : %s\n' ...
            'Iter        : %d\n' ...
            'Objective   : %.6f  (= 1 - F1, lower is better)\n' ...
            'F1 at iter  : %.6f\n' ...
            'Best F1     : %.6f  (best objective so far = %.6f)\n' ...
            'Iter runtime: %.1f sec\n' ...
            '\n' ...
            'HPs at iter %d:\n%s\n' ...
            'Best HPs so far:\n%s\n'], ...
            run_label, n, last_obj, 1 - last_obj, ...
            1 - best_obj, best_obj, last_runtm, ...
            n, evalc('disp(last_X)'), evalc('disp(best_X)'));

        % Pipe body via a temp file to avoid quoting headaches with
        % multi-line bodies / brackets / special chars in HP value strings.
        tmpf = [tempname() '_bo_email.txt'];
        fid  = fopen(tmpf, 'w');
        fprintf(fid, '%s', body);
        fclose(fid);

        % Strip any quotes/backticks that snuck into run_label or HP
        % strings before they reach the shell. Subject is the only
        % surface that has to survive shell quoting.
        subj_safe = regexprep(subj, '[`"$\\]', '_');
        cmd = sprintf('cat "%s" | mail -s "%s" %s ; rm -f "%s"', ...
            tmpf, subj_safe, email_addr, tmpf);
        [ret, out] = system(cmd);
        if ret ~= 0
            warning('email_bayesopt_outputfcn:sendFailed', ...
                'mail send returned %d: %s', ret, strtrim(out));
        end
    catch ME
        % Never let a mail problem kill a multi-hour BO run.
        warning('email_bayesopt_outputfcn:exception', ...
            'email_bayesopt_outputfcn failed (suppressed): %s', ME.message);
    end
end
