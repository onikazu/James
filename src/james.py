# 引数にエピソード数入れる

# TODO 変数の読み込み保存


import player11
import threading
import numpy as np
import sys
import os
import random
import pickle
from memory import Memory
from actor import Actor
from critic import Critic


class James(player11.Player11, threading.Thread):
    def __init__(self):
        super(James, self).__init__()
        self.name = "James"
        self.m_strCommand = ""
        self.command_step = 0

        # =============command log ===============
        self.kick_num = 0
        self.turn_num = 0
        self.dash_num = 0

        # =============for reinforcement learning=================
        # pretrainか検査
        if sys.argv[2] == "pretrain":
            self.pre_flag = True
            self.max_number_of_steps = 100
        else:
            self.pre_flag = False
            self.max_number_of_steps = 1000

        # actionについて(すべてのコマンドイチ変数にまとめてしまおう（強制的に一次元に）)
        # 563の行動空間要素数
        self.action = random.randint(0, 663)
        self.actions = ("turn", "kick", "dash")

        # DDPG で用いるオブジェクトの読み込み
        actor_file = open("./A_C_M/actor_file", "rb")
        critic_file = open("./A_C_M/critic_file", "rb")
        memory_file = open("./A_C_M/memory_file", "rb")
        self.critic = pickle.load(critic_file)
        self.actor = pickle.load(actor_file)
        self.memory = pickle.load(memory_file)
        actor_file.close()
        critic_file.close()
        memory_file.close()

        # reward(エピソードごと、ステップごと)
        self.episode_reward = 0
        self.reward = 0
        # 1試行のstep数
        self.num_this_episode = int(sys.argv[3])

        # 状態空間を表す変数
        self.state = [0, 0, 0, 0, 0, 0]
        self.next_state = [0, 0, 0, 0, 0, 0]

    def analyzeMessage(self, message):
        """
        メッセージの解析
        :param message:
        :return:
        """
        # 初期メッセージの処理
        # print("p11:message:", message)
        if message.startswith("(init "):
            self.analyzeInitialMessage(message)
        # 視覚メッセージの処理
        elif message.startswith("(see "):
            self.analyzeVisualMessage(message)
        # 体調メッセージの処理
        elif message.startswith("(sense_body "):
            self.analyzePhysicalMessage(message)

            # ====================================================================
            if self.pre_flag:
                # コマンドの決定
                self.play_0()
                if self.command_step <= 1:
                    self.first_step_flow()
                else:
                    # 前のstepによる変化を読み取る
                    self.recognize_nextstate()
                    # 報酬の計算
                    self.calc_reward()
                    # メモリの保存
                    self.add_memory()
                    # 状態の経過
                    self.state = self.next_state
                # コマンドの送信
                self.send(self.m_strCommand)
                # ステップの進行　エピソード終了の検査
                # play中のコマンド送信ならばstepを進める
                if self.m_strPlayMode.startswith("play_on"):
                    self.command_step += 1
                self.check_episode_end()
            # ====================================================================
            else:
                if self.command_step <= 1:
                    self.play_0()
                    self.first_step_flow()
                else:
                    self.recognize_nextstate()
                    self.calc_reward()
                    self.add_memory()
                    self.episode_reward += self.reward
                    states, actions, rewards, next_states = self.memory.sample(20)
                    next_actions = self.actor.get_actions(next_states)
                    next_qs = self.critic.get_qs(next_states, next_actions)
                    loss, q = self.critic.train(states, actions, rewards, next_qs)
                    action_gradients = self.critic.get_action_gradients(states, actions)
                    self.actor.train(states, action_gradients[0])
                    action = self.actor.get_action_for_train(self.state, self.num_this_episode)
                    self.play_0()
        # 聴覚メッセージの処理
        elif message.startswith("(hear "):
            self.analyzeAuralMessage(message)
        # サーバパラメータの処理
        elif message.startswith("(server_param"):
            self.analyzeServerParam(message)
        # プレーヤーパラメータの処理
        elif message.startswith("(player_param"):
            self.analyzePlayerParam(message)
        # プレーヤータイプの処理
        # elif message.startswith("(player_type"):
        #     self.analyzePlayerType(message)
        #     # print("player_type_message", message)
        # think 処理
        elif message.startswith("(think"):
            self.send("(done)")
        # エラーの処理
        else:
            print("p11 サーバーからエラーが伝えられた:", message)
            print("p11 エラー発生原因のコマンドは右記の通り :", self.m_strCommand)

    # 実行
    def play_0(self):
        """
        コマンドの決定
        :return:
        """
        # キックオフ前？
        if self.checkInitialMode():
            if self.checkInitialMode():
                self.setKickOffPosition()
                command = \
                    "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                self.m_strCommand = command
        # (コマンド生成)===================
        # pretrain modeなら
        if sys.argv[2] == "pretrain":
            # ランダムにアクションを決定
            self.action = random.randint(0, 562)
            if 0 <= self.action <= 360:
                self.m_strCommand = "(turn {})".format(self.action - 180)
            if 361 <= self.action <= 461:
                self.m_strCommand = "(dash {})".format(self.action - 361)
            if 462 <= self.action <= 662:
                self.m_strCommand = "(kick {})".format(self.action - 562)


    # ===============================

    def first_step_flow(self):
        self.state = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY, self.m_dNeck, self.m_dStamina]
        self.calc_reward()


    def recognize_nextstate(self):
        if self.command_step > 1:
            self.nextstate = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY, self.m_dNeck, self.m_dStamina]

    def calc_reward(self):
        self.reward = 0
        # 右チーム
        if self.m_strSide.startswith("r"):
            # ゴールすれば
            if self.m_strPlayMode == "(goal_r)":
                self.reward += 1000

        # 左チーム
        if self.m_strSide.startswith("l"):
            # ゴールすれば
            if self.m_strPlayMode == "(goal_l)":
                self.reward += 1000

        # 共通
        # ボールをキックできれば
        if self.getDistance(self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY) <= 0.7 and \
                self.m_strCommand.startswith("(kick"):
            self.reward += 1

    def add_memory(self):
        self.memory.add((self.state, self.action, self.reward, self.next_state))

    def check_episode_end(self):
        """
        1エピソードが終わったかどうか判定し終わっていたら報酬をリセットし、次のエピソードを始める準備をする
        :return:
        """

        # stepが規定step(1000 or 100step) に達したか？
        if self.command_step % self.max_number_of_steps == 0 and self.command_step != 0:
            print("{}episode finished".format(self.num_this_episode))
            # self.total_reward_vec = np.hstack((self.total_reward_vec[1:], self.episode_reward))  # 報酬を記録
            # # ログの保存
            # with open("./logs/{0}_{1}_reward.log".format(self.m_strTeamName, self.m_iNumber), "a") as the_file:
            #     the_file.write("{0},{1}\n".format(self.num_this_episode, self.episode_reward))
            #
            # # コマンドログの保存
            # with open("./logs/{0}_{1}_command.log".format(self.m_strTeamName, self.m_iNumber), "a") as the_file:
            #     the_file.write(
            #         "{0},{1},{2},{3}\n".format(self.num_this_episode, self.kick_num, self.turn_num, self.dash_num))
            # 変数の保存
            actor_file = open("./A_C_M/actor_file", "wb")
            critic_file = open("./A_C_M/critic_file", "wb")
            memory_file = open("./A_C_M/memory_file", "wb")
            pickle.dump(self.actor, actor_file)
            pickle.dump(self.critic, critic_file)
            pickle.dump(self.memory, memory_file)
            actor_file.close()
            critic_file.close()
            memory_file.close()


if __name__ == "__main__":
    plays = []
    for i in range(22):
        p = James()
        plays.append(p)
        teamname = str(p.__class__.__name__)
        if i < 11:
            teamname += "left"
        else:
            teamname += "right"
        plays[i].initialize((i % 11 + 1), teamname, "localhost", 6000)
        plays[i].start()

# 離散化させなくてはならない？(6分割**5変数の状態が生み出される)
# 状態s一覧
#
# self.m_dX
# self.m_dY
# self.m_dNeck
# self.m_dBallX
# self.m_dBallY
#
# 行動a一覧
# (turn 60)
# (turn -60)
# (dash 100)
# (dash -100)
# (kick 100 0)
# (kick 50 0)
